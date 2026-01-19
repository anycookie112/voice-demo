"""
Local Whisper Real-Time-ish STT Adapter
"""

import asyncio
import contextlib
import re
from typing import AsyncIterator, Optional, Any

import numpy as np
import torch
import whisper

from events import STTEvent, STTOutputEvent

class WhisperPytorchSTT:
    def __init__(
        self,
        api_key: Optional[str] = None,
        format_turns: bool = True,
        
        # Whisper Settings
        model_size: str = "base",
        sample_rate: int = 16000,
        device: str = "cuda",
        language: str = None, 
        
        # VAD / Sensitivity Settings
        silence_threshold: float = 2000.0, 
        min_silence_chunks: int = 4,       
        logprob_threshold: float = -1.0,   
        no_speech_threshold: float = 0.4,  # <-- LOWERED: More aggressive at detecting silence (default 0.6)
        
        **kwargs: Any,
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.min_silence_chunks = min_silence_chunks
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.language = language

        # State for deduplication
        self._last_text = ""

        # Device Logic
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device

        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._close_signal = asyncio.Event()

        print(f"[INFO] Loading Whisper '{model_size}' on {self.device}...")
        try:
            self._model = whisper.load_model(model_size, device=self.device)
            print(f"[INFO] Model loaded. Language: {self.language if self.language else 'Auto'}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise e

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def receive_events(self) -> AsyncIterator[STTEvent]:
        buffer = bytearray()
        silence_counter = 0
        chunk_size = 3200  # ~0.2s at 16kHz, 16-bit mono (tune as needed)
        partial_emit_interval = 2  # emit partial every N chunks
        chunk_counter = 0
        while True:
            try:
                if self._close_signal.is_set() and self._audio_queue.empty():
                    break
                chunk = await self._audio_queue.get()
            except asyncio.CancelledError:
                break

            if chunk is None:
                if buffer:
                    # Final partial before output
                    partial = await self._transcribe_async(bytes(buffer))
                    if partial and partial.strip():
                        from events import STTChunkEvent
                        yield STTChunkEvent.create(partial)
                    text = partial
                    if self._is_valid_output(text):
                        from events import STTOutputEvent
                        yield STTOutputEvent.create(text)
                break

            buffer.extend(chunk)
            chunk_counter += 1

            # Emit partials at intervals for real-time feedback
            if chunk_counter % partial_emit_interval == 0 and len(buffer) > 0:
                partial = await self._transcribe_async(bytes(buffer))
                if partial and partial.strip():
                    from events import STTChunkEvent
                    yield STTChunkEvent.create(partial)

            # Energy-based VAD
            if self._is_silence(chunk):
                silence_counter += 1
            else:
                silence_counter = 0

            # Process buffer when silence is detected
            if silence_counter >= self.min_silence_chunks and len(buffer) > 0:
                pcm = bytes(buffer)
                buffer.clear()
                silence_counter = 0
                chunk_counter = 0

                # Emit a final partial before output
                partial = await self._transcribe_async(pcm)
                if partial and partial.strip():
                    from events import STTChunkEvent
                    yield STTChunkEvent.create(partial)
                text = partial
                # Yield only if valid and not a duplicate
                if self._is_valid_output(text):
                    from events import STTOutputEvent
                    yield STTOutputEvent.create(text)
                # After a turn, just reset state and keep listening for new audio

    async def send_audio(self, audio_chunk: bytes) -> None:
        if not self._close_signal.is_set():
            await self._audio_queue.put(audio_chunk)

    async def close(self) -> None:
        self._close_signal.set()
        with contextlib.suppress(asyncio.QueueFull):
            await self._audio_queue.put(None)

    # -------------------------------------------------------------------------
    # Internal Logic
    # -------------------------------------------------------------------------

    async def _ensure_connection(self): pass

    def _is_silence(self, audio_chunk: bytes) -> bool:
        if not audio_chunk: return True
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        if samples.size == 0: return True
        
        rms = np.sqrt(np.mean(samples**2))
        return rms < self.silence_threshold

    async def _transcribe_async(self, pcm_bytes: bytes) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_blocking, pcm_bytes)

    def _transcribe_blocking(self, pcm_bytes: bytes) -> str:
        if not pcm_bytes: return ""

        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        use_fp16 = (self.device == "cuda")

        try:
            result = self._model.transcribe(
                audio_np,
                language=self.language,
                fp16=use_fp16,
                beam_size=1,             # <--- CHANGED from 5 to 1 (5x Speedup)
                best_of=1,               # <--- Ensure no parallel sampling
                temperature=0.0,         # <--- Greedy decoding
                no_speech_threshold=self.no_speech_threshold, 
                logprob_threshold=self.logprob_threshold,
                condition_on_previous_text=False
            )
            return result.get("text", "").strip()

        except Exception as e:
            print(f"[ERROR] Transcribe error: {e}")
            return ""

    def _is_valid_output(self, text: str) -> bool:
        """
        Validates text against hallucinations and duplicates.
        """
        cleaned = text.strip()
        if not cleaned:
            return False

        # 1. Check Blocklist (Case insensitive)
        # These are standard hallucinations from the training data
        blocklist = {
            "we'll be right back", 
            "we'll be right back.",
            "you", "you.", "thank you.", "thanks.", 
            "mbc news", "bye.", "bye", 
            "字幕", "字幕 by", "谢谢", "谢谢观看", "不客气",
            "copyright", "all rights reserved",
            "uncaptioned"
        }
        
        # Remove punctuation for the check to catch "We'll be right back!"
        check_str = re.sub(r'[^\w\s]', '', cleaned).lower()
        if check_str in blocklist or cleaned.lower() in blocklist:
            return False

        # 2. Check for bracketed noise like [silence]
        if re.match(r'^\[.*?\]$', cleaned) or re.match(r'^\(.*?\)$', cleaned):
            return False

        # 3. Deduplication (Prevent same phrase looping)
        # If the exact same text comes out twice in a row, block the second one.
        if cleaned == self._last_text:
            return False
            
        self._last_text = cleaned
        return True