# import asyncio
# import logging
# import re
# import numpy as np
# from typing import AsyncIterator, Optional
# from faster_whisper import WhisperModel
# from events import STTEvent, STTOutputEvent

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("LocalWhisperSTT")

# class LocalWhisperSTT:
#     def __init__(
#         self,
#         model_size: str = "large-v3", 
#         sample_rate: int = 16000,
#         device: str = "cuda",         
#         compute_type: str = "float16",
#         # FIX 1: Increase threshold (try 500-1000). 
#         # If your mic is sensitive, 100 picks up background hiss.
#         silence_threshold: float = 600.0, 
        
#         # FIX 2: Increase wait time. 
#         # 1.2 seconds allows for a natural "thinking pause" in conversation.
#         min_silence_duration: float = 1.2, 
        
#         # FIX 3: Increase min audio. 
#         # Ignore sounds shorter than 0.5s (like a cough or chair squeak).
#         min_audio_duration: float = 0.5,   
        
#         # FIX 4: Safety valve.
#         max_buffer_duration: float = 15.0, 
#     ):
#         self.sample_rate = sample_rate
#         self.silence_threshold = silence_threshold
#         self.device = device
#         self.compute_type = compute_type
        
#         # Save these settings (Fixes the NameError)
#         self.min_audio_duration = min_audio_duration
#         self.max_buffer_duration = max_buffer_duration
#         self.min_silence_seconds = min_silence_duration

#         # Calculate bytes per second (16-bit mono = 2 bytes per sample)
#         self.bytes_per_second = sample_rate * 2 
        
#         self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
#         self._close_signal = asyncio.Event()

#         logger.info(f"Loading Whisper model '{model_size}' on {self.device}...")
#         self._model = WhisperModel(
#             model_size,
#             device=self.device,
#             compute_type=self.compute_type,
#         )
#         logger.info("Whisper model loaded.")

#     async def receive_events(self) -> AsyncIterator[STTEvent]:
#         """
#         Drop-in replacement for AssemblyAI.receive_events()
#         Instead of reading from a WebSocket, we read from the local audio queue.
#         """
#         buffer = bytearray()
        
#         # State tracking for VAD
#         has_speech = False 
#         silence_start_time = None
        
#         logger.info("Local STT listening...")

#         while not self._close_signal.is_set():
#             try:
#                 # Wait for next audio chunk
#                 chunk = await self._audio_queue.get()
#             except asyncio.CancelledError:
#                 break

#             # Sentinel: None means stream is finishing
#             if chunk is None:
#                 # Flush remaining audio if there was speech in it
#                 if buffer and has_speech:
#                     text = await self._transcribe_async(bytes(buffer))
#                     if text: yield STTOutputEvent.create(text)
#                 break

#             # 1. Add chunk to buffer
#             buffer.extend(chunk)

#             # 2. Analyze Audio Energy (VAD)
#             rms = self._calculate_rms(chunk)
#             is_silent = rms < self.silence_threshold
            
#             # Debugging (Optional: Uncomment to tune sensitivity)
#             # logger.info(f"RMS: {rms}")

#             current_time = asyncio.get_running_loop().time()

#             if not is_silent:
#                 # Speech Detected
#                 has_speech = True
#                 silence_start_time = None # Reset silence timer
#             else:
#                 # Silence Detected
#                 if silence_start_time is None:
#                     silence_start_time = current_time # Start silence timer
            
#             # 3. Decision Logic: Should we transcribe now?
#             buffer_duration = len(buffer) / self.bytes_per_second
#             silence_duration = (current_time - silence_start_time) if silence_start_time else 0.0

#             should_transcribe = False
            
#             # Condition A: Normal Sentence End
#             # We have audio, we heard speech, and we've had enough silence after the speech
#             if (buffer_duration >= self.min_audio_duration and 
#                 has_speech and 
#                 silence_duration >= self.min_silence_seconds):
#                 should_transcribe = True
                
#             # Condition B: Buffer Overflow (Safety Valve)
#             # User has been talking too long without pausing
#             elif buffer_duration >= self.max_buffer_duration:
#                 should_transcribe = True

#             # 4. Transcribe Execution
#             if should_transcribe:
#                 # Edge Case: If buffer is full but it was ALL silence, drop it.
#                 if not has_speech:
#                     buffer.clear()
#                     silence_start_time = current_time
#                     continue

#                 # Prepare audio data
#                 pcm_data = bytes(buffer)
                
#                 # Reset state immediately so we can capture next phrase
#                 buffer.clear()
#                 has_speech = False
#                 silence_start_time = None 

#                 # Run Inference (Non-blocking)
#                 text = await self._transcribe_async(pcm_data)
                
#                 # Yield the event (This matches AssemblyAI's OutputEvent)
#                 if text:
#                     yield STTOutputEvent.create(text)

#     async def send_audio(self, audio_chunk: bytes) -> None:
#         """Queue audio for processing."""
#         await self._audio_queue.put(audio_chunk)

#     async def close(self) -> None:
#         """Stop processing."""
#         self._close_signal.set()
#         await self._audio_queue.put(None)

#     # -------------------------------------------------------------------------
#     # Internal Helpers
#     # -------------------------------------------------------------------------

#     def _calculate_rms(self, audio_chunk: bytes) -> float:
#         if not audio_chunk: return 0.0
#         samples = np.frombuffer(audio_chunk, dtype=np.int16)
#         if samples.size == 0: return 0.0
#         # Cast to float32 to prevent overflow
#         sq = samples.astype(np.float32) ** 2
#         return float(np.sqrt(np.mean(sq)))

#     async def _transcribe_async(self, pcm_bytes: bytes) -> str:
#         loop = asyncio.get_running_loop()
#         return await loop.run_in_executor(None, self._transcribe_blocking, pcm_bytes)

#     def _transcribe_blocking(self, pcm_bytes: bytes) -> str:
#         audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

#         try:
#             # 1. Start Transcription
#             # Change '_' to 'info' to catch language data
#             segments, info = self._model.transcribe(
#                 audio,
#                 beam_size=1,
#                 without_timestamps=True,
#                 condition_on_previous_text=False,
#                 # Bias it slightly anyway
#                 initial_prompt="English, Malay, Mandarin, Cantonese", 
#                 vad_filter=True,
#                 vad_parameters=dict(min_silence_duration_ms=800)
#             )

#             # 2. Define Allowed Languages
#             # 'en' = English
#             # 'ms' = Malay
#             # 'zh' = Chinese (Mandarin & Cantonese)
#             # 'yue' = Cantonese (Specific code, but Whisper usually defaults to 'zh')
#             allowed_langs = {'en', 'ms', 'zh', 'yue'}

#             # 3. Check Detected Language
#             if info.language not in allowed_langs:
#                 print(f"Ignored language: {info.language} (Confidence: {info.language_probability})")
#                 return "" # Return empty if it's French/Spanish/etc.

#             # 4. Return Text
#             return " ".join([segment.text for segment in segments])

#         except Exception as e:
#             print(f"Transcription error: {e}")
#             return ""

#     def _filter_hallucinations(self, text: str) -> str:
#         if not text: return ""
#         clean = text.strip()
        
#         # Block common Whisper hallucinations
#         blocklist = {
#             "you", "You.", "you.", "You", 
#             "No, no, no, no, no.", "No, no, no.", "no no no",
#             "Thank you.", "Thanks.", 
#             "MBC News", "Amara.org", "Subtitle by"
#         }
        
#         if clean in blocklist:
#             return ""

#         # Block repeats "word word word word"
#         if re.search(r'\b(\w+)( \1){2,}', clean, re.IGNORECASE):
#             return ""

#         return clean



import asyncio
import logging
import re
import numpy as np
from typing import AsyncIterator, Optional, Tuple
from faster_whisper import WhisperModel
from events import STTEvent, STTOutputEvent
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImprovedWhisperSTT")


class LocalWhisperSTT:
    """
    Enhanced Whisper STT with:
    - Rolling energy window for better noise detection
    - Adaptive silence threshold
    - Minimum speech duration before transcription
    - Better hallucination filtering
    - End-of-turn detection with confirmation window
    """
    
    def __init__(
        self,
        model_size: str = "large-v3",
        sample_rate: int = 16000,
        device: str = "cuda",
        compute_type: str = "float16",
        
        # === NOISE/SILENCE DETECTION ===
        # Base threshold - will be adaptive
        base_silence_threshold: float = 800.0,
        
        # Rolling window for energy analysis (helps ignore single spikes)
        energy_window_size: int = 5,  # Look at last 5 chunks
        
        # Percentage of window that must be "loud" to count as speech
        speech_ratio_threshold: float = 0.6,  # 60% of window must be loud
        
        # === TIMING PARAMETERS ===
        # Wait this long after speech stops before transcribing
        end_of_speech_silence: float = 1.0,
        
        # Additional silence to confirm turn is really over
        end_of_turn_silence: float = 0.5,
        
        # Minimum duration of speech to transcribe (filters coughs, clicks)
        min_speech_duration: float = 0.8,
        
        # Maximum buffer before force transcription
        max_buffer_duration: float = 15.0,
        
        # === NOISE REDUCTION ===
        # Use spectral gating (requires noisereduce library)
        use_noise_reduction: bool = False,
        
        # Calibration: measure noise floor from first N seconds
        noise_calibration_duration: float = 2.0,
    ):
        self.sample_rate = sample_rate
        self.base_silence_threshold = base_silence_threshold
        self.device = device
        self.compute_type = compute_type
        
        # Energy window for smoothing
        self.energy_window_size = energy_window_size
        self.speech_ratio_threshold = speech_ratio_threshold
        self.energy_window = deque(maxlen=energy_window_size)
        
        # Timing parameters
        self.end_of_speech_silence = end_of_speech_silence
        self.end_of_turn_silence = end_of_turn_silence
        self.min_speech_duration = min_speech_duration
        self.max_buffer_duration = max_buffer_duration
        
        # Noise reduction
        self.use_noise_reduction = use_noise_reduction
        self.noise_calibration_duration = noise_calibration_duration
        self.noise_profile = None
        self.is_calibrated = False
        self.calibration_buffer = bytearray()
        
        # Adaptive threshold
        self.adaptive_threshold = base_silence_threshold
        self.background_noise_samples = deque(maxlen=100)
        
        # Bytes per second (16-bit mono)
        self.bytes_per_second = sample_rate * 2
        
        # Queues and events
        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._close_signal = asyncio.Event()
        
        # Load Whisper model
        logger.info(f"Loading Whisper model '{model_size}' on {self.device}...")
        self._model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("Whisper model loaded.")
        
        # Try to import noise reduction library
        if self.use_noise_reduction:
            try:
                import noisereduce as nr
                self.nr = nr
                logger.info("Noise reduction enabled (noisereduce library found)")
            except ImportError:
                logger.warning("noisereduce library not found. Install with: pip install noisereduce")
                self.use_noise_reduction = False

    async def receive_events(self) -> AsyncIterator[STTEvent]:
        """Process audio with improved noise handling and turn detection"""
        buffer = bytearray()
        
        # Speech state tracking
        has_speech = False
        speech_start_time = None
        silence_start_time = None
        last_speech_time = None
        
        # Track continuous speech duration
        speech_chunks_count = 0
        
        logger.info("🎤 Improved STT listening... (Calibrating noise floor)")
        
        while not self._close_signal.is_set():
            try:
                chunk = await self._audio_queue.get()
            except asyncio.CancelledError:
                break
            
            # Sentinel: end of stream
            if chunk is None:
                if buffer and has_speech:
                    text = await self._transcribe_async(bytes(buffer))
                    if text:
                        yield STTOutputEvent.create(text)
                break
            
            # === NOISE CALIBRATION PHASE ===
            if not self.is_calibrated:
                self.calibration_buffer.extend(chunk)
                calibration_duration = len(self.calibration_buffer) / self.bytes_per_second
                
                if calibration_duration >= self.noise_calibration_duration:
                    self._calibrate_noise_floor(bytes(self.calibration_buffer))
                    self.calibration_buffer.clear()
                    logger.info("✓ Noise floor calibrated. Ready for speech.")
                continue
            
            # === MAIN PROCESSING ===
            buffer.extend(chunk)
            current_time = asyncio.get_running_loop().time()
            
            # Calculate energy with rolling window
            chunk_rms = self._calculate_rms(chunk)
            self.energy_window.append(chunk_rms)
            
            # Determine if this is speech or noise
            is_speech = self._is_speech_in_window()
            
            # Update background noise estimate during silence
            if not is_speech:
                self.background_noise_samples.append(chunk_rms)
                self._update_adaptive_threshold()
            
            # === STATE MACHINE ===
            if is_speech:
                # Speech detected
                if not has_speech:
                    # Transition: Silence -> Speech
                    speech_start_time = current_time
                    has_speech = True
                    speech_chunks_count = 0
                    logger.debug("🗣️  Speech started")
                
                speech_chunks_count += 1
                last_speech_time = current_time
                silence_start_time = None
                
            else:
                # Silence detected
                if has_speech and silence_start_time is None:
                    # Transition: Speech -> Silence
                    silence_start_time = current_time
                    logger.debug("🤫 Silence after speech")
            
            # === TRANSCRIPTION TRIGGERS ===
            buffer_duration = len(buffer) / self.bytes_per_second
            
            # Calculate durations
            speech_duration = (last_speech_time - speech_start_time) if (last_speech_time and speech_start_time) else 0
            silence_duration = (current_time - silence_start_time) if silence_start_time else 0
            
            should_transcribe = False
            transcribe_reason = ""
            
            # Trigger 1: Natural end of utterance
            if (has_speech and 
                speech_duration >= self.min_speech_duration and
                silence_duration >= self.end_of_speech_silence):
                should_transcribe = True
                transcribe_reason = "end_of_utterance"
            
            # Trigger 2: Clear end of turn (longer silence)
            elif (has_speech and
                  speech_duration >= self.min_speech_duration and
                  silence_duration >= (self.end_of_speech_silence + self.end_of_turn_silence)):
                should_transcribe = True
                transcribe_reason = "end_of_turn"
            
            # Trigger 3: Safety valve (buffer overflow)
            elif buffer_duration >= self.max_buffer_duration and has_speech:
                should_transcribe = True
                transcribe_reason = "buffer_overflow"
            
            # === EXECUTE TRANSCRIPTION ===
            if should_transcribe:
                # Ignore if we didn't actually hear enough speech
                if speech_duration < self.min_speech_duration:
                    logger.debug(f"⏭️  Skipped: too short ({speech_duration:.2f}s)")
                    buffer.clear()
                    has_speech = False
                    speech_start_time = None
                    silence_start_time = None
                    continue
                
                logger.info(f"📝 Transcribing ({transcribe_reason}): {speech_duration:.2f}s speech, {silence_duration:.2f}s silence")
                
                pcm_data = bytes(buffer)
                
                # Reset state
                buffer.clear()
                has_speech = False
                speech_start_time = None
                silence_start_time = None
                speech_chunks_count = 0
                
                # Transcribe
                text = await self._transcribe_async(pcm_data)
                
                if text:
                    yield STTOutputEvent.create(text)

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Queue audio for processing"""
        await self._audio_queue.put(audio_chunk)

    async def close(self) -> None:
        """Stop processing"""
        self._close_signal.set()
        await self._audio_queue.put(None)

    # =========================================================================
    # NOISE DETECTION & CALIBRATION
    # =========================================================================
    
    def _calibrate_noise_floor(self, audio_bytes: bytes):
        """Measure ambient noise level from initial silence"""
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = samples.astype(np.float32) / 32768.0
        
        # Calculate noise characteristics
        rms_values = []
        chunk_size = self.sample_rate // 10  # 100ms chunks
        
        for i in range(0, len(audio_float), chunk_size):
            chunk = audio_float[i:i + chunk_size]
            if len(chunk) > 0:
                rms = np.sqrt(np.mean(chunk ** 2))
                rms_values.append(rms * 32768)  # Convert back to int16 scale
        
        if rms_values:
            # Use 90th percentile to avoid outliers
            noise_level = np.percentile(rms_values, 90)
            # Set threshold at 2.5x the noise floor
            self.adaptive_threshold = max(noise_level * 2.5, self.base_silence_threshold)
            logger.info(f"📊 Noise floor: {noise_level:.1f}, Threshold: {self.adaptive_threshold:.1f}")
        
        # Store noise profile for spectral reduction if enabled
        if self.use_noise_reduction:
            self.noise_profile = audio_float
        
        self.is_calibrated = True
    
    def _is_speech_in_window(self) -> bool:
        """Check if rolling window contains enough loud samples to be speech"""
        if len(self.energy_window) < 2:
            return False
        
        # Count how many samples exceed threshold
        loud_count = sum(1 for rms in self.energy_window if rms > self.adaptive_threshold)
        loud_ratio = loud_count / len(self.energy_window)
        
        return loud_ratio >= self.speech_ratio_threshold
    
    def _update_adaptive_threshold(self):
        """Continuously adjust threshold based on background noise"""
        if len(self.background_noise_samples) < 10:
            return
        
        # Use recent background samples to adjust threshold
        recent_noise = np.percentile(list(self.background_noise_samples), 75)
        
        # Smooth adjustment: 80% old + 20% new
        target_threshold = max(recent_noise * 2.5, self.base_silence_threshold)
        self.adaptive_threshold = 0.8 * self.adaptive_threshold + 0.2 * target_threshold
        
        # Debug: uncomment to monitor adaptation
        # logger.debug(f"Adaptive threshold: {self.adaptive_threshold:.1f}")
    
    def _calculate_rms(self, audio_chunk: bytes) -> float:
        """Calculate RMS energy of audio chunk"""
        if not audio_chunk:
            return 0.0
        samples = np.frombuffer(audio_chunk, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        sq = samples.astype(np.float32) ** 2
        return float(np.sqrt(np.mean(sq)))

    # =========================================================================
    # TRANSCRIPTION
    # =========================================================================
    
    async def _transcribe_async(self, pcm_bytes: bytes) -> str:
        """Non-blocking transcription"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_blocking, pcm_bytes)
    
    def _transcribe_blocking(self, pcm_bytes: bytes) -> str:
        """Blocking transcription with noise reduction and filtering"""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Apply noise reduction if enabled
        if self.use_noise_reduction and self.noise_profile is not None:
            try:
                audio = self.nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    y_noise=self.noise_profile,
                    stationary=True,
                )
            except Exception as e:
                logger.warning(f"Noise reduction failed: {e}")
        
        try:
            segments, info = self._model.transcribe(
                audio,
                beam_size=5,  # Increased for better quality
                without_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt="English, Malay, Mandarin, Cantonese.",
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    threshold=0.5,
                    min_speech_duration_ms=250,
                ),
                # Language detection
                language=None,  # Auto-detect
            )
            
            # Filter by allowed languages
            allowed_langs = {'en', 'ms', 'zh', 'yue'}
            if info.language not in allowed_langs:
                logger.debug(f"⏭️  Ignored language: {info.language} ({info.language_probability:.2f})")
                return ""
            
            # Combine segments
            texts = [s.text.strip() for s in segments if s.text.strip()]
            if not texts:
                return ""
            
            raw_text = " ".join(texts)
            
            # Filter hallucinations and noise
            filtered_text = self._filter_hallucinations(raw_text)
            
            if filtered_text:
                logger.info(f"✓ [{info.language}] {filtered_text}")
            
            return filtered_text
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return ""
    
    def _filter_hallucinations(self, text: str) -> str:
        """Filter common Whisper hallucinations and noise artifacts"""
        if not text:
            return ""
        
        clean = text.strip()
        
        # Block common hallucinations
        blocklist = {
            "you", "you.", "You", "You.",
            "thank you", "thank you.", "thanks", "thanks.",
            "bye", "bye.", "goodbye",
            # Common subtitle artifacts
            "MBC News", "Amara.org", "Subtitle by",
            # Single letters or numbers
            "a", "i", "1", "2",
            # Breathing sounds transcribed as words
            "huh", "uh", "um", "hmm", "mhm",
        }
        
        if clean.lower() in blocklist:
            return ""
        
        # Block if too short (likely noise)
        if len(clean) < 3:
            return ""
        
        # Block repeated words: "word word word"
        if re.search(r'\b(\w+)( \1){2,}', clean, re.IGNORECASE):
            return ""
        
        # Block if all punctuation/numbers (no letters)
        if not re.search(r'[a-zA-Z\u4e00-\u9fff]', clean):
            return ""
        
        # Block excessive punctuation
        if clean.count('.') > 5 or clean.count(',') > 8:
            return ""
        
        return clean