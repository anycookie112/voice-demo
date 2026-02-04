"""
Qwen3 Text-to-Speech Adapter

Qwen3-TTS model for text-to-speech synthesis.

Input: text via send_text()
Output: TTSChunkEvent (PCM 16-bit, 24000 Hz) via receive_events()
"""

import asyncio
import contextlib
from typing import AsyncIterator, Optional
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

try:
    from events import TTSChunkEvent
except ImportError:
    class TTSChunkEvent:
        @staticmethod
        def create(audio_chunk: bytes):
            return type("Event", (), {"audio": audio_chunk})()


class Qwen3TTS:
    """Qwen-3 TTS model wrapper for generating speech from text."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        language: str = "auto",  # Supported: auto, chinese, english, french, german, italian, japanese, korean, portuguese, russian, spanish
        speaker: str = "Vivian",
        instruct: Optional[str] = None,
        sample_rate: int = 24000,
        chunk_ms: int = 50,
    ):
        """
        Initialize Qwen3-TTS adapter.
        
        Args:
            model_path: Path to the Qwen3-TTS model
            device: Device to run model on (cuda:0, cpu, etc.)
            dtype: Model data type
            attn_implementation: Attention implementation type
            language: Language for synthesis (Auto, Chinese, English)
            speaker: Speaker voice name
            instruct: Optional instruction for voice style
            sample_rate: Output audio sample rate
            chunk_ms: Chunk size in milliseconds for streaming
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Normalize mpx typo to mps
        if device.lower() == "mpx":
            device = "mps"
        
        # Validate mps availability
        if device == "mps" and not torch.backends.mps.is_available():
            print("[Qwen3TTS] MPS not available. Falling back to CPU.")
            device = "cpu"
        
        self.device = device
        self.model_path = model_path
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.language = language
        self.speaker = speaker
        self.instruct = instruct
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms

        # Text queue: send_text() pushes here, receive_events() consumes
        self._text_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._close_signal = asyncio.Event()
        self._interrupt_signal = asyncio.Event()
        
        print(f"[Qwen3TTS] Loading model from {model_path} on {device}...")
        self._model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            local_files_only=True,
            trust_remote_code=True,
        )
        print(f"[Qwen3TTS] Model loaded successfully!")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_instruct(self, instruct: str) -> None:
        """
        Set the voice instruction/tone for TTS synthesis.
        
        Args:
            instruct: Instruction string describing how to speak (e.g., "Speak warmly and friendly")
        """
        if instruct and instruct.strip():
            self.instruct = instruct.strip()
            print(f"[Qwen3TTS] Voice instruction set: {self.instruct[:50]}...")

    async def send_text(self, text: Optional[str]) -> None:
        """
        Queue text for synthesis.
        """
        if text is None:
            return

        # Send empty string as a "flush"/no-op, but don't synthesize audio
        if text == "":
            await self._text_queue.put("")  # marker (no audio)
            return

        if not text.strip():
            return

        await self._text_queue.put(text)

    async def receive_events(self) -> AsyncIterator[object]:
        """
        Async generator yielding TTSChunkEvent objects with PCM audio chunks, and TTSEndEvent at the end of each turn.
        """
        from events import TTSEndEvent
        
        while not self._close_signal.is_set():
            # Clear interrupt signal at the start of each new synthesis
            self._interrupt_signal.clear()
            
            # Wait for next text to synthesize, or break if closed
            try:
                text = await asyncio.wait_for(self._text_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if self._close_signal.is_set():
                    break
                continue

            if self._close_signal.is_set():
                break

            # Empty string used as a "flush"/turn marker: skip audio
            if text is None or text == "":
                continue

            # Check for interrupt before starting synthesis
            if self._interrupt_signal.is_set():
                print("[Qwen3TTS] Skipping synthesis due to interrupt")
                continue

            # Run Qwen3 TTS in a threadpool to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            pcm_bytes = await loop.run_in_executor(
                None, self._synthesize_to_pcm_bytes, text
            )

            # Check for interrupt after synthesis (before streaming)
            if self._interrupt_signal.is_set():
                print("[Qwen3TTS] Discarding synthesized audio due to interrupt")
                continue

            # Stream chunks as TTSChunkEvent
            bytes_per_sample = 2  # int16
            samples_per_chunk = int(self.sample_rate * self.chunk_ms / 1000)
            bytes_per_chunk = samples_per_chunk * bytes_per_sample

            for i in range(0, len(pcm_bytes), bytes_per_chunk):
                # Check for interrupt during streaming
                if self._close_signal.is_set() or self._interrupt_signal.is_set():
                    if self._interrupt_signal.is_set():
                        print("[Qwen3TTS] Stopping audio stream due to interrupt")
                    break

                chunk = pcm_bytes[i : i + bytes_per_chunk]
                if not chunk:
                    break

                yield TTSChunkEvent.create(chunk)

            # Only emit TTSEndEvent if we weren't interrupted
            if not self._interrupt_signal.is_set():
                yield TTSEndEvent.create()
                print("[DEBUG] Qwen3TTS: Turn complete (TTSEndEvent emitted)")
            else:
                print("[Qwen3TTS] Turn interrupted - no TTSEndEvent emitted")

    async def close(self) -> None:
        """
        Signal the adapter to stop and cleanup.
        """
        if self._close_signal.is_set():
            return  # Already closed
        self._close_signal.set()
        
        # Clear any pending text in the queue
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Put a sentinel to unblock any pending queue get()
        with contextlib.suppress(asyncio.QueueFull):
            await self._text_queue.put(None)
        
        # Cleanup model
        if hasattr(self, '_model') and self._model is not None:
            del self._model
            self._model = None
        
        # Clear CUDA cache to free memory
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("[Qwen3TTS] CUDA cache cleared")
        except Exception as e:
            print(f"[Qwen3TTS] CUDA cleanup error: {e}")

    def interrupt(self) -> None:
        """
        Interrupt current TTS playback immediately.
        Called when user starts speaking during TTS output.
        """
        print("[Qwen3TTS] Interrupt signal received - stopping current playback")
        self._interrupt_signal.set()
        # Clear any pending text in the queue
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _synthesize_to_pcm_bytes(self, text: str) -> bytes:
        """
        Blocking: run Qwen3 TTS on text and return PCM16 bytes at sample_rate.
        """
        try:
            # Generate audio using Qwen3-TTS
            wavs, sr = self._model.generate_custom_voice(
                text=text,
                language=self.language,
                speaker=self.speaker,
                instruct=self.instruct if self.instruct else None,
            )
            
            print(f"[Qwen3TTS] Model output sample rate: {sr}, target: {self.sample_rate}")
            
            if wavs is None or len(wavs) == 0:
                print("[Qwen3TTS] No audio generated")
                return b""
            
            # Get the first audio sample
            audio = wavs[0]
            
            # Ensure it's a numpy array
            if torch.is_tensor(audio):
                audio = audio.cpu().numpy()
            
            # Flatten if needed
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            print(f"[Qwen3TTS] Audio shape: {audio.shape}, min: {audio.min():.3f}, max: {audio.max():.3f}")
            
            # Resample if needed
            if sr != self.sample_rate:
                print(f"[Qwen3TTS] Resampling from {sr} to {self.sample_rate}")
                audio = self._resample(audio, sr, self.sample_rate)
            
            # Convert to PCM16
            return self._float32_to_pcm16(audio)
            
        except Exception as e:
            print(f"[Qwen3TTS] Error during synthesis: {e}")
            import traceback
            traceback.print_exc()
            return b""

    @staticmethod
    def _float32_to_pcm16(audio: np.ndarray) -> bytes:
        """
        Convert float32 [-1, 1] waveform to 16-bit PCM bytes.
        """
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767.0).astype(np.int16)
        return audio_int16.tobytes()

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio using scipy for better quality.
        Falls back to linear interpolation if scipy is not available.
        """
        if orig_sr == target_sr:
            return audio

        try:
            # Use scipy for high-quality resampling
            from scipy import signal
            new_length = int(len(audio) * target_sr / orig_sr)
            resampled = signal.resample(audio, new_length)
            return resampled.astype(np.float32)
        except ImportError:
            # Fallback to linear interpolation
            duration = audio.shape[0] / float(orig_sr)
            new_length = int(duration * target_sr)

            t_orig = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
            t_new = np.linspace(0.0, duration, num=new_length, endpoint=False)

            return np.interp(t_new, t_orig, audio).astype(np.float32)

    async def synthesize_to_wav(self, text: str, path: str = "test_qwen3.wav") -> None:
        """
        Helper method to synthesize text directly to a WAV file (for testing).
        """
        import wave
        
        loop = asyncio.get_running_loop()
        pcm_bytes = await loop.run_in_executor(None, self._synthesize_to_pcm_bytes, text)

        with wave.open(path, "wb") as f:
            f.setnchannels(1)          # mono
            f.setsampwidth(2)          # 16-bit
            f.setframerate(self.sample_rate)
            f.writeframes(pcm_bytes)

        print(f"[Qwen3TTS] Wrote {len(pcm_bytes)} bytes to {path}")
    

