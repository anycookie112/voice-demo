import asyncio
import os
import threading
import logging
import numpy as np
from typing import AsyncIterator, Optional
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download

# Assuming TTSChunkEvent is imported from your events module
try:
    from events import TTSChunkEvent
except ImportError:
    class TTSChunkEvent:
        @staticmethod
        def create(audio_chunk: bytes):
            return type("Event", (), {"audio": audio_chunk})()

from demo.vibevoice_tts_main import StreamingTTSService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VibeVoiceAsync")

class VibeVoiceAsyncTTS:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        voice_preset: str = None, 
        # --- QUALITY SETTINGS ---
        inference_steps: int = 60,    # INCREASED: 5 -> 30 for high quality
        temperature: float = 0.3,     # DECREASED: 0.7 -> 0.3 for stability
        cfg_scale: float = 4.0,       # INCREASED: 1.5 -> 4.0 for better voice adherence
        hf_repo_id: str = "vibevoice/VibeVoice-7B",
    ):
        self.model_path = model_path
        self.device = device
        self.voice_key = voice_preset
        
        # Store quality parameters
        self.inference_steps = inference_steps
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        
        self._ensure_model_downloaded(model_path, hf_repo_id)
        
        logger.info(f"Initializing VibeVoice (Steps={inference_steps}, Temp={temperature})...")
        self.service = StreamingTTSService(
            model_path=model_path,
            device=device,
            inference_steps=inference_steps
        )
        self.service.load()
        
        if not hasattr(self.service, 'sample_rate'):
            self.service.sample_rate = 24000 
            
        logger.info("VibeVoice Model loaded.")

        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()
        self._stop_event = threading.Event()
        self._processing_task: Optional[asyncio.Task] = None
        
        # Buffer for sentence-based processing
        self._text_buffer = ""

        self._processing_task = asyncio.create_task(self._generation_worker())

    def _ensure_model_downloaded(self, local_path: str, repo_id: str):
        if not os.path.exists(local_path):
            logger.warning(f"Downloading {repo_id} to {local_path}...")
            snapshot_download(repo_id=repo_id, local_dir=local_path, local_dir_use_symlinks=False)

    async def send_text(self, text: Optional[str]) -> None:
        """
        Smart Buffering: Accumulates text and only sends to GPU 
        when a sentence is complete. This improves intonation significantly.
        """
        if text is None: return
        if not text.strip(): return

        # Append incoming text to buffer
        self._text_buffer += text

        # Check for sentence endings, including Chinese punctuation
        delimiters = ['.', '?', '!', '\n', '。', '！', '？']

        while any(d in self._text_buffer for d in delimiters):
            # Find the earliest delimiter
            earliest_idx = len(self._text_buffer)
            for d in delimiters:
                idx = self._text_buffer.find(d)
                if idx != -1 and idx < earliest_idx:
                    earliest_idx = idx

            # Extract the full sentence (including the punctuation)
            sentence = self._text_buffer[:earliest_idx+1].strip()
            self._text_buffer = self._text_buffer[earliest_idx+1:]

            if sentence:
                await self._input_queue.put(sentence)

    async def flush_buffer(self):
        """Force process remaining text (e.g., at end of stream)"""
        if self._text_buffer.strip():
            await self._input_queue.put(self._text_buffer.strip())
            self._text_buffer = ""

    async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
        try:
            while True:
                chunk = await self._output_queue.get()
                if chunk is None: break
                if isinstance(chunk, Exception):
                    logger.error(f"Generation Error: {chunk}")
                    continue
                yield TTSChunkEvent.create(chunk)
                self._output_queue.task_done()
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        # Process any remaining text in buffer before closing
        await self.flush_buffer()
        
        self._stop_event.set()
        await self._input_queue.put(None)
        if self._processing_task:
            await self._processing_task
        logger.info("Service closed.")

    async def _generation_worker(self):
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        while not self._stop_event.is_set():
            text = await self._input_queue.get()
            if text is None: break

            # Process valid text
            if text.strip():
                await loop.run_in_executor(executor, self._run_sync_stream, text, loop)
            
            self._input_queue.task_done()

        await self._output_queue.put(None)
        executor.shutdown(wait=False)

    def _run_sync_stream(self, text: str, loop: asyncio.AbstractEventLoop):
        try:
            # Use the instance variables we set in __init__
            stream_iterator = self.service.stream(
                text=text,
                voice_key=self.voice_key,
                inference_steps=self.inference_steps, # 30 steps
                temperature=self.temperature,         # 0.3
                cfg_scale=self.cfg_scale              # 4.0
            )

            for chunk_numpy in stream_iterator:
                if self._stop_event.is_set(): break
                
                if hasattr(self.service, 'chunk_to_pcm16'):
                    pcm_bytes = self.service.chunk_to_pcm16(chunk_numpy)
                else:
                    pcm_bytes = (chunk_numpy * 32767).astype(np.int16).tobytes()

                loop.call_soon_threadsafe(self._output_queue.put_nowait, pcm_bytes)

        except Exception as e:
            loop.call_soon_threadsafe(self._output_queue.put_nowait, e)


# import asyncio
# import os
# import threading
# import logging
# from typing import AsyncIterator, Optional
# from concurrent.futures import ThreadPoolExecutor

# # Import Hugging Face Hub for auto-downloading
# from huggingface_hub import snapshot_download

# # Assuming TTSChunkEvent is imported from your events module
# try:
#     from events import TTSChunkEvent
# except ImportError:
#     class TTSChunkEvent:
#         @staticmethod
#         def create(audio_chunk: bytes):
#             return type("Event", (), {"audio": audio_chunk})()

# # Import your local VibeVoice class
# from demo.vibevoice_tts_main import StreamingTTSService

# # Configure Logger
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("VibeVoiceAsync")

# class VibeVoiceAsyncTTS:
#     def __init__(
#         self,
#         model_path: str,
#         device: str = "cuda",
#         voice_preset: str = None, 
#         inference_steps: int = 5,
#         hf_repo_id: str = "microsoft/VibeVoice-Realtime-0.5B", # Default official repo
#     ):
#         """
#         Async wrapper for VibeVoiceTTS.
#         Auto-downloads model/tokenizer if model_path does not exist.
#         """
#         self.model_path = model_path
#         self.device = device
#         self.voice_key = voice_preset
#         self.inference_steps = inference_steps
        
#         # 1. Check and Download Model + Tokenizer if missing
#         self._ensure_model_downloaded(model_path, hf_repo_id)
        
#         # 2. Initialize the synchronous VibeVoice service
#         logger.info(f"Initializing VibeVoice service from {model_path}...")
#         try:
#             self.service = StreamingTTSService(
#                 model_path=model_path,
#                 device=device,
#                 inference_steps=inference_steps
#             )
#             self.service.load()
#         except Exception as e:
#             logger.error(f"Failed to load VibeVoice service: {e}")
#             raise e
        
#         # 3. Ensure sample_rate exists
#         if not hasattr(self.service, 'sample_rate'):
#             self.service.sample_rate = 24000 
            
#         logger.info("VibeVoice Model loaded successfully.")

#         # 4. Setup Async/Sync Bridge
#         self._input_queue = asyncio.Queue()
#         self._output_queue = asyncio.Queue()
#         self._stop_event = threading.Event()
#         self._processing_task: Optional[asyncio.Task] = None
        
#         # Start the background worker
#         self._processing_task = asyncio.create_task(self._generation_worker())

#     def _ensure_model_downloaded(self, local_path: str, repo_id: str):
#         """
#         Checks if the local directory exists and is populated.
#         If not, downloads the VibeVoice model (includes Qwen tokenizer config) from HF.
#         """
#         # Check if folder exists
#         if not os.path.exists(local_path):
#             logger.warning(f"Model path '{local_path}' not found. Downloading {repo_id}...")
#             try:
#                 snapshot_download(
#                     repo_id=repo_id,
#                     local_dir=local_path,
#                     local_dir_use_symlinks=False, # Copy actual files so it persists cleanly
#                     ignore_patterns=["*.git*"] # Clean download
#                 )
#                 logger.info(f"Download complete. Model saved to {local_path}")
#             except Exception as e:
#                 logger.error(f"Failed to download model from Hugging Face: {e}")
#                 raise RuntimeError(
#                     f"Model not found at {local_path} and download failed. "
#                     "Please check your internet connection or HF token."
#                 ) from e
#         else:
#             # Optional: Check if the folder is empty
#             if not os.listdir(local_path):
#                 logger.warning(f"Directory '{local_path}' is empty. Downloading {repo_id}...")
#                 snapshot_download(
#                     repo_id=repo_id,
#                     local_dir=local_path,
#                     local_dir_use_symlinks=False
#                 )

#     async def send_text(self, text: Optional[str]) -> None:
#         if text is None: return
#         if not text.strip(): return
#         await self._input_queue.put(text)

#     async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
#         try:
#             while True:
#                 chunk = await self._output_queue.get()
                
#                 if chunk is None: 
#                     break
                
#                 if isinstance(chunk, Exception):
#                     logger.error(f"Generation Error: {chunk}")
#                     continue

#                 yield TTSChunkEvent.create(chunk)
#                 self._output_queue.task_done()
                
#         except asyncio.CancelledError:
#             pass

#     async def close(self) -> None:
#         self._stop_event.set()
#         await self._input_queue.put(None)
#         if self._processing_task:
#             await self._processing_task
#         logger.info("Service closed.")

#     async def _generation_worker(self):
#         loop = asyncio.get_running_loop()
#         executor = ThreadPoolExecutor(max_workers=1)

#         while not self._stop_event.is_set():
#             text = await self._input_queue.get()
#             if text is None: break

#             await loop.run_in_executor(
#                 executor, 
#                 self._run_sync_stream, 
#                 text, 
#                 loop
#             )
#             self._input_queue.task_done()

#         await self._output_queue.put(None)
#         executor.shutdown(wait=False)

#     def _run_sync_stream(self, text: str, loop: asyncio.AbstractEventLoop):
#         try:
#             stream_iterator = self.service.stream(
#                 text=text,
#                 voice_key=self.voice_key,
#                 inference_steps=self.inference_steps,
#                 temperature=0.7,
#                 cfg_scale=1.5
#             )

#             for chunk_numpy in stream_iterator:
#                 if self._stop_event.is_set(): break
                
#                 # Convert Numpy -> PCM Bytes
#                 if hasattr(self.service, 'chunk_to_pcm16'):
#                     pcm_bytes = self.service.chunk_to_pcm16(chunk_numpy)
#                 else:
#                     # Fallback if method missing: convert float32 [-1,1] to int16 bytes
#                     pcm_bytes = (chunk_numpy * 32767).astype(np.int16).tobytes()

#                 loop.call_soon_threadsafe(
#                     self._output_queue.put_nowait, 
#                     pcm_bytes
#                 )

#         except Exception as e:
#             loop.call_soon_threadsafe(
#                 self._output_queue.put_nowait, 
#                 e
#             )