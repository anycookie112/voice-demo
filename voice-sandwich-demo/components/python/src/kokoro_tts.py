# """
# ElevenLabs Text-to-Speech Streaming

# Python implementation of ElevenLabs streaming TTS API.
# Converts text to PCM audio in real-time using WebSocket streaming.

# Input: Text strings
# Output: TTS events (tts_chunk for audio chunks)
# """

# import asyncio
# import base64
# import contextlib
# import json
# import os
# from typing import AsyncIterator, Optional

# import websockets
# from websockets.client import WebSocketClientProtocol

# from events import TTSChunkEvent


# from kokoro import KPipeline
# from IPython.display import display, Audio
# import soundfile as sf
# import torch
# pipeline = KPipeline(lang_code='a')
# text = '''
# [Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
# '''
# generator = pipeline(text, voice='af_heart')
# for i, (gs, ps, audio) in enumerate(generator):
#     print(i, gs, ps)
#     display(Audio(data=audio, rate=24000, autoplay=i==0))
#     sf.write(f'{i}.wav', audio, 24000)




# class KokoroTTS:
#     _ws: Optional[WebSocketClientProtocol]
#     _connection_signal: asyncio.Event
#     _close_signal: asyncio.Event

#     def __init__(self,
#         voice = 'af_heart',
#         model = 'kokoro_tts_v1'):

#         self.voice = voice
#         self.model = model
#         self._ws = None
#         self._connection_signal = asyncio.Event()
#         self._close_signal = asyncio.Event()

        

"""
Kokoro Text-to-Speech Adapter (local)

Drop-in replacement for ElevenLabsTTS, but using a local Kokoro model.

Input: text via send_text()
Output: TTSChunkEvent (PCM 16-bit, 24000 Hz) via receive_events()
"""

import asyncio
import contextlib
from typing import AsyncIterator, Optional, List

import numpy as np

from kokoro import KPipeline  # pip install kokoro>=0.9.4
from events import TTSChunkEvent



# Global Pipeline Cache (Loaded once per process)
_GLOBAL_PIPELINES = {}

class KokoroTTS:
    _close_signal: asyncio.Event

    def __init__(
        self,
        lang_code: str = "a",          # 'a' = American English, 'b' = British, etc.
        voice: str = "af_heart",       # see Kokoro VOICES.md
        sample_rate: int = 24000,
        chunk_ms: int = 50,            # how big each streamed audio chunk is
        speed: float = 1.0,
    ):
        """
        Local Kokoro-based TTS adapter, streaming PCM in small chunks.

        Args:
            lang_code: Kokoro language code ('a' = American English, ...)
            voice: Kokoro voice name (e.g. 'af_heart', 'af_bella', ...)
            sample_rate: Audio sample rate (Kokoro uses 24000 Hz by default)
            chunk_ms: chunk size in milliseconds for streaming
            speed: playback speed multiplier (if supported by pipeline)
        """
        self.lang_code = lang_code
        self.voice = voice
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.speed = speed

        # Text queue: send_text() pushes here, receive_events() consumes
        self._text_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._close_signal = asyncio.Event()

        # Check global cache
        self._ensure_pipeline(self.lang_code)

    def _ensure_pipeline(self, lang_code: str):
        """Ensure pipeline for language exists in GLOBAL cache."""
        if lang_code not in _GLOBAL_PIPELINES:
            print(f"[Kokoro] Loading pipeline for '{lang_code}' (Global Cache)...")
            try:
                _GLOBAL_PIPELINES[lang_code] = KPipeline(lang_code=lang_code)
                print(f"[Kokoro] Successfully loaded pipeline for '{lang_code}'")
            except Exception as e:
                print(f"[Kokoro] Error loading pipeline for '{lang_code}': {e}. Falling back to 'a'.")
                import traceback
                traceback.print_exc()
                if 'a' not in _GLOBAL_PIPELINES:
                     _GLOBAL_PIPELINES['a'] = KPipeline(lang_code='a')
        else:
             print(f"[Kokoro] Using cached pipeline for '{lang_code}'")

    @property
    def _pipeline(self):
        # Helper to get current pipeline from cache
        return _GLOBAL_PIPELINES.get(self.lang_code) or _GLOBAL_PIPELINES.get('a')

    def set_language(self, lang_code: str):
        """Switch the active TTS language."""
        if lang_code == self.lang_code:
            return
            
        # Basic validation/mapping if needed
        # Kokoro codes: a(US), b(UK), j(Japanese), z(Chinese), etc.
        valid_codes = ['a', 'b', 'j', 'z', 'e', 'f', 'h', 'i', 'p'] 
        
        if lang_code not in valid_codes:
            print(f"[Kokoro] Ignored unsupported language code: {lang_code}")
            return
            
        print(f"[Kokoro] Switching language: {self.lang_code} -> {lang_code}")
        self.lang_code = lang_code
        self._ensure_pipeline(self.lang_code)

        if lang_code == 'z' and self.voice == 'af_heart':
             self.voice = 'zf_xiaobei' # Example switch to a Chinese voice
             print(f"[Kokoro] Switched to Chinese voice: {self.voice}")
        elif lang_code == 'a' and self.voice == 'zf_xiaobei':
             self.voice = 'af_heart'
             print(f"[Kokoro] Switched to English voice: {self.voice}")

    # -------------------------------------------------------------------------
    # Public API (same as ElevenLabsTTS)
    # -------------------------------------------------------------------------

    async def send_text(self, text: Optional[str]) -> None:
        """
        Queue text for synthesis. Mimics ElevenLabsTTS.send_text().
        """
        if text is None:
            return

        # Keep behavior similar to ElevenLabs:
        # - send empty string as a "flush"/no-op, but don't synthesize audio
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
                # You could emit a "turn complete" signal here if your system needs it
                continue

            # Run Kokoro TTS in a threadpool to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            pcm_bytes = await loop.run_in_executor(
                None, self._synthesize_to_pcm_bytes, text
            )

            # Stream chunks as TTSChunkEvent, similar to ElevenLabs streaming
            bytes_per_sample = 2  # int16
            samples_per_chunk = int(self.sample_rate * self.chunk_ms / 1000)
            bytes_per_chunk = samples_per_chunk * bytes_per_sample

            for i in range(0, len(pcm_bytes), bytes_per_chunk):
                if self._close_signal.is_set():
                    break

                chunk = pcm_bytes[i : i + bytes_per_chunk]
                if not chunk:
                    break

                yield TTSChunkEvent.create(chunk)

                # Optional: simulate "real-time" pacing
                # await asyncio.sleep(self.chunk_ms / 1000.0)

            # Emit TTSEndEvent after all audio chunks for this turn
            yield TTSEndEvent.create()
            print("[DEBUG] Kokoro: Turn complete (TTSEndEvent emitted)")

    async def close(self) -> None:
        """
        Signal the adapter to stop. Mimics ElevenLabsTTS.close().
        """
        self._close_signal.set()
        # Put a sentinel to unblock any pending queue get()
        with contextlib.suppress(asyncio.QueueFull):
            await self._text_queue.put(None)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _synthesize_to_pcm_bytes(self, text: str) -> bytes:
        """
        Blocking: run Kokoro pipeline on text and return PCM16 bytes at sample_rate.
        """
        audio_segments: List[np.ndarray] = []
        
        # Use our property to get pipeline
        pipeline = self._pipeline
        if not pipeline:
            print("[Kokoro] No pipeline available, skipping synthesis.")
            return b""
            
        generator = pipeline(
            text, 
            voice=self.voice,
            speed=self.speed, 
            split_pattern=r'\n+'
        )

        for i, (gs, ps, audio) in enumerate(generator):
             # audio is float32 numpy array, typically 24000Hz (or model default)
             audio_segments.append(audio)
             
        if not audio_segments:
            return b""

        full_audio = np.concatenate(audio_segments, axis=0)

        # Resample from Kokoro’s native 24kHz → requested sample_rate
        if self.sample_rate != 24000:
            full_audio = self._resample(full_audio, 24000, self.sample_rate)

        return self._float32_to_pcm16(full_audio)

    @staticmethod
    def _float32_to_pcm16(audio: np.ndarray) -> bytes:
        """
        Convert float32 [-1, 1] waveform to 16-bit PCM bytes.
        """
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767.0).astype(np.int16)
        return audio_int16.tobytes()


    # inside KokoroTTS class, for testing only
    async def synthesize_to_wav(self, text: str, path: str = "test_kokoro.wav") -> None:
        import wave
        import asyncio

        loop = asyncio.get_running_loop()
        pcm_bytes = await loop.run_in_executor(None, self._synthesize_to_pcm_bytes, text)

        with wave.open(path, "wb") as f:
            f.setnchannels(1)          # mono
            f.setsampwidth(2)          # 16-bit
            f.setframerate(self.sample_rate)  # 24000
            f.writeframes(pcm_bytes)

        print(f"[DEBUG] Wrote {len(pcm_bytes)} bytes to {path}")


    import numpy as np

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio

        duration = audio.shape[0] / float(orig_sr)
        new_length = int(duration * target_sr)

        t_orig = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        t_new = np.linspace(0.0, duration, num=new_length, endpoint=False)

        return np.interp(t_new, t_orig, audio).astype(np.float32)
