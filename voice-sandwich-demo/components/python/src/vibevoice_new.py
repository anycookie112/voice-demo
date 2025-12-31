# """
# VibeVoice Text-to-Speech Streaming

# Python implementation of VibeVoice 1.5b local TTS with proper streaming.
# Converts text to PCM audio in real-time using async/sync bridge pattern.

# Input: Text strings
# Output: TTS events (tts_chunk for audio chunks)
# """

# import asyncio
# import os
# import threading
# import logging
# import numpy as np
# from typing import AsyncIterator, Optional
# from concurrent.futures import ThreadPoolExecutor
# import torch

# from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
# from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
# from vibevoice.modular.lora_loading import load_lora_assets

# try:
#     from events import TTSChunkEvent
# except ImportError:
#     class TTSChunkEvent:
#         @staticmethod
#         def create(audio_chunk: bytes):
#             return type("Event", (), {"audio": audio_chunk})()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("VibeVoiceTTS")


# class VibeVoiceTTS:
#     """VibeVoice TTS class for streaming text-to-speech generation"""
    
#     def __init__(
#         self,
#         model_path: str = "microsoft/VibeVoice-1.5b",
#         voice_sample_path: Optional[str] = None,
#         device: Optional[str] = None,
#         checkpoint_path: Optional[str] = None,
#         cfg_scale: float = 1.3,
#         num_inference_steps: int = 10,
#         disable_prefill: bool = False,
#         seed: Optional[int] = None,
#         chunk_size: int = 2400,  # Number of samples per chunk (0.1s at 24kHz)
#     ):
#         """
#         Initialize VibeVoice TTS with streaming support
        
#         Args:
#             model_path: Path to the HuggingFace model directory
#             voice_sample_path: Path to voice sample WAV file for cloning
#             device: Device for inference (cuda/mps/cpu). Auto-detect if None
#             checkpoint_path: Path to fine-tuned LoRA checkpoint (optional)
#             cfg_scale: Classifier-Free Guidance scale (default: 1.3)
#             num_inference_steps: Number of DDPM inference steps (default: 10)
#             disable_prefill: Disable voice cloning/prefill
#             seed: Random seed for reproducibility
#             chunk_size: Audio samples per chunk for streaming (default: 2400 = 0.1s)
#         """
#         # Auto-detect device
#         if device is None:
#             if torch.cuda.is_available():
#                 device = "cuda"
#             elif torch.backends.mps.is_available():
#                 device = "mps"
#             else:
#                 device = "cpu"
        
#         # Normalize mpx typo to mps
#         if device.lower() == "mpx":
#             device = "mps"
        
#         # Validate mps availability
#         if device == "mps" and not torch.backends.mps.is_available():
#             logger.warning("MPS not available. Falling back to CPU.")
#             device = "cpu"
        
#         self.device = device
#         self.model_path = model_path
#         self.voice_sample_path = voice_sample_path
#         self.checkpoint_path = checkpoint_path
#         self.cfg_scale = cfg_scale
#         self.num_inference_steps = num_inference_steps
#         self.disable_prefill = disable_prefill
#         self.chunk_size = chunk_size
#         self.sample_rate = 24000  # VibeVoice uses 24kHz
        
#         # Set seed if provided
#         if seed is not None:
#             torch.manual_seed(seed)
#             if torch.cuda.is_available():
#                 torch.cuda.manual_seed_all(seed)
        
#         # Determine dtype and attention implementation
#         if self.device == "mps":
#             self.load_dtype = torch.float32
#             self.attn_impl = "sdpa"
#         elif self.device == "cuda":
#             self.load_dtype = torch.bfloat16
#             self.attn_impl = "flash_attention_2"
#         else:  # cpu
#             self.load_dtype = torch.float32
#             self.attn_impl = "sdpa"
        
#         logger.info(f"VibeVoice TTS initialized on device: {self.device}")
#         logger.info(f"Using dtype: {self.load_dtype}, attention: {self.attn_impl}")
        
#         # Initialize model state
#         self._model = None
#         self._processor = None
        
#         # Async/threading infrastructure for streaming
#         self._input_queue = asyncio.Queue()
#         self._output_queue = asyncio.Queue()
#         self._stop_event = threading.Event()
#         self._processing_task: Optional[asyncio.Task] = None
        
#         # Text buffer for sentence-based processing
#         self._text_buffer = ""
        
#         # Load model synchronously on init
#         self._load_model()
        
#         # Start background generation worker
#         # Will be started when receive_events is called
    
#     def _load_model(self):
#         """Load model and processor (blocking)"""
#         if self._model is not None:
#             return
        
#         logger.info(f"Loading VibeVoice model from {self.model_path}...")
        
#         # Load processor
#         self._processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        
#         # Load model with device-specific logic
#         try:
#             if self.device == "mps":
#                 self._model = VibeVoiceForConditionalGenerationInference.from_pretrained(
#                     self.model_path,
#                     torch_dtype=self.load_dtype,
#                     attn_implementation=self.attn_impl,
#                     device_map=None,
#                 )
#                 self._model.to("mps")
#             elif self.device == "cuda":
#                 self._model = VibeVoiceForConditionalGenerationInference.from_pretrained(
#                     self.model_path,
#                     torch_dtype=self.load_dtype,
#                     device_map="cuda",
#                     attn_implementation=self.attn_impl,
#                 )
#             else:  # cpu
#                 self._model = VibeVoiceForConditionalGenerationInference.from_pretrained(
#                     self.model_path,
#                     torch_dtype=self.load_dtype,
#                     device_map="cpu",
#                     attn_implementation=self.attn_impl,
#                 )
#         except Exception as e:
#             if self.attn_impl == 'flash_attention_2':
#                 logger.warning(f"Error with flash_attention_2: {e}")
#                 logger.info("Falling back to SDPA attention...")
#                 self._model = VibeVoiceForConditionalGenerationInference.from_pretrained(
#                     self.model_path,
#                     torch_dtype=self.load_dtype,
#                     device_map=(self.device if self.device in ("cuda", "cpu") else None),
#                     attn_implementation='sdpa'
#                 )
#                 if self.device == "mps":
#                     self._model.to("mps")
#             else:
#                 raise e
        
#         # Load checkpoint if provided
#         if self.checkpoint_path:
#             logger.info(f"Loading checkpoint from {self.checkpoint_path}...")
#             load_lora_assets(self._model, self.checkpoint_path)
        
#         self._model.eval()
#         self._model.set_ddpm_inference_steps(num_steps=self.num_inference_steps)
        
#         logger.info("Model loaded successfully!")
    
#     async def send_text(self, text: Optional[str]) -> None:
#         """
#         Queue text for generation with smart sentence buffering
        
#         Args:
#             text: Text to synthesize. None or empty string signals end of input.
#         """
#         if text is None:
#             return
        
#         if not text.strip():
#             return
        
#         # Just queue the text - sentence detection happens in the worker
#         await self._input_queue.put(text)
    
#     async def flush_buffer(self):
#         """Force process remaining text (e.g., at end of stream)"""
#         if self._text_buffer.strip():
#             await self._input_queue.put(("FLUSH", self._text_buffer.strip()))
#             self._text_buffer = ""
    
#     async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
#         """
#         Receive audio chunk events as they're generated
        
#         Yields:
#             TTSChunkEvent: Audio chunks as they become available
#         """
#         # Start generation task if not already running
#         if self._processing_task is None:
#             self._processing_task = asyncio.create_task(self._generation_worker())
        
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
    
#     async def _generation_worker(self):
#         """Main generation loop that processes queued text"""
#         loop = asyncio.get_running_loop()
#         executor = ThreadPoolExecutor(max_workers=1)
        
#         while not self._stop_event.is_set():
#             try:
#                 text = await self._input_queue.get()
                
#                 if text is None:
#                     break
                
#                 # Handle flush command
#                 if isinstance(text, tuple) and text[0] == "FLUSH":
#                     text = text[1]
                
#                 # Process valid text
#                 if text.strip():
#                     await loop.run_in_executor(
#                         executor,
#                         self._run_sync_stream,
#                         text,
#                         loop
#                     )
                
#                 self._input_queue.task_done()
                
#             except Exception as e:
#                 logger.error(f"Error in generation worker: {e}")
#                 loop.call_soon_threadsafe(
#                     self._output_queue.put_nowait,
#                     e
#                 )
        
#         await self._output_queue.put(None)
#         executor.shutdown(wait=False)
    
#     def _run_sync_stream(self, text: str, loop: asyncio.AbstractEventLoop):
#         """
#         Synchronous generation with streaming output
        
#         Args:
#             text: Text to synthesize
#             loop: Event loop for thread-safe callback
#         """
#         try:
#             # Format text with speaker label (required by VibeVoice processor)
#             if not text.strip().startswith("Speaker"):
#                 formatted_text = f"Speaker 1: {text}"
#             else:
#                 formatted_text = text
            
#             # Prepare voice samples
#             voice_samples = [self.voice_sample_path] if self.voice_sample_path else []
            
#             inputs = self._processor(
#                 text=[formatted_text],
#                 voice_samples=[voice_samples] if voice_samples else None,
#                 padding=True,
#                 return_tensors="pt",
#                 return_attention_mask=True,
#             )
            
#             # Move to device
#             target_device = self.device if self.device != "cpu" else "cpu"
#             for k, v in inputs.items():
#                 if torch.is_tensor(v):
#                     inputs[k] = v.to(target_device)
            
#             # Generate
#             with torch.no_grad():
#                 outputs = self._model.generate(
#                     **inputs,
#                     max_new_tokens=None,
#                     cfg_scale=self.cfg_scale,
#                     tokenizer=self._processor.tokenizer,
#                     generation_config={'do_sample': False},
#                     verbose=False,
#                     is_prefill=not self.disable_prefill,
#                 )
            
#             # Extract and stream audio in chunks
#             if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
#                 audio = outputs.speech_outputs[0]
                
#                 # Convert to numpy
#                 if torch.is_tensor(audio):
#                     # Convert bfloat16 to float32 first if needed
#                     if audio.dtype == torch.bfloat16:
#                         audio = audio.float()
#                     audio = audio.cpu().numpy()
                
#                 # Flatten audio
#                 audio = audio.flatten()
                
#                 # Stream in chunks
#                 total_samples = len(audio)
#                 for i in range(0, total_samples, self.chunk_size):
#                     if self._stop_event.is_set():
#                         break
                    
#                     chunk = audio[i:i + self.chunk_size]
                    
#                     # Convert to PCM16 bytes
#                     chunk_int16 = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
#                     pcm_bytes = chunk_int16.tobytes()
                    
#                     # Send to output queue thread-safely
#                     loop.call_soon_threadsafe(
#                         self._output_queue.put_nowait,
#                         pcm_bytes
#                     )
            
#         except Exception as e:
#             logger.error(f"Error in sync stream: {e}")
#             loop.call_soon_threadsafe(
#                 self._output_queue.put_nowait,
#                 e
#             )
    
#     async def close(self) -> None:
#         """Clean up resources"""
#         # Process any remaining text in buffer before closing
#         await self.flush_buffer()
        
#         self._stop_event.set()
#         await self._input_queue.put(None)
        
#         if self._processing_task:
#             try:
#                 await self._processing_task
#             except asyncio.CancelledError:
#                 pass
        
#         # Clear model from memory
#         if self._model is not None:
#             del self._model
#             self._model = None
        
#         if self._processor is not None:
#             del self._processor
#             self._processor = None
        
#         # Clear CUDA cache if using GPU
#         if self.device == "cuda":
#             torch.cuda.empty_cache()
        
#         logger.info("VibeVoice TTS closed")
"""
VibeVoice Text-to-Speech Streaming

Python implementation of VibeVoice 1.5b local TTS with proper streaming.
Converts text to PCM audio in real-time using async/sync bridge pattern.

Input: Text strings
Output: TTS events (tts_chunk for audio chunks)
"""

import asyncio
import os
import threading
import logging
import numpy as np
from typing import AsyncIterator, Optional
from concurrent.futures import ThreadPoolExecutor
import torch

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.lora_loading import load_lora_assets

try:
    from events import TTSChunkEvent
except ImportError:
    class TTSChunkEvent:
        @staticmethod
        def create(audio_chunk: bytes):
            return type("Event", (), {"audio": audio_chunk})()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VibeVoiceTTS")


class VibeVoiceTTS:
    """VibeVoice TTS class for streaming text-to-speech generation"""
    
    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-1.5b",
        voice_sample_path: Optional[str] = None,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        cfg_scale: float = 1.3,
        num_inference_steps: int = 20,
        disable_prefill: bool = False,
        seed: Optional[int] = None,
        chunk_size: int = 2400,  # Number of samples per chunk (0.1s at 24kHz)
    ):
        """
        Initialize VibeVoice TTS with streaming support
        
        Args:
            model_path: Path to the HuggingFace model directory
            voice_sample_path: Path to voice sample WAV file for cloning
            device: Device for inference (cuda/mps/cpu). Auto-detect if None
            checkpoint_path: Path to fine-tuned LoRA checkpoint (optional)
            cfg_scale: Classifier-Free Guidance scale (default: 1.3)
            num_inference_steps: Number of DDPM inference steps (default: 10)
            disable_prefill: Disable voice cloning/prefill
            seed: Random seed for reproducibility
            chunk_size: Audio samples per chunk for streaming (default: 2400 = 0.1s)
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
            logger.warning("MPS not available. Falling back to CPU.")
            device = "cpu"
        
        self.device = device
        self.model_path = model_path
        self.voice_sample_path = voice_sample_path
        self.checkpoint_path = checkpoint_path
        self.cfg_scale = cfg_scale
        self.num_inference_steps = num_inference_steps
        self.disable_prefill = disable_prefill
        self.chunk_size = chunk_size
        self.sample_rate = 24000  # VibeVoice uses 24kHz
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Determine dtype and attention implementation
        if self.device == "mps":
            self.load_dtype = torch.float32
            self.attn_impl = "sdpa"
        elif self.device == "cuda":
            self.load_dtype = torch.bfloat16
            self.attn_impl = "flash_attention_2"
        else:  # cpu
            self.load_dtype = torch.float32
            self.attn_impl = "sdpa"
        
        logger.info(f"VibeVoice TTS initialized on device: {self.device}")
        logger.info(f"Using dtype: {self.load_dtype}, attention: {self.attn_impl}")
        
        # Initialize model state
        self._model = None
        self._processor = None
        
        # Async/threading infrastructure for streaming
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()
        self._stop_event = threading.Event()
        self._processing_task: Optional[asyncio.Task] = None
        
        # Text buffer for sentence-based processing
        self._text_buffer = ""
        
        # Load model synchronously on init
        self._load_model()
        
        # Start background generation worker
        # Will be started when receive_events is called
    
    def _load_model(self):
        """Load model and processor (blocking)"""
        if self._model is not None:
            return
        
        logger.info(f"Loading VibeVoice model from {self.model_path}...")
        
        # Determine if this is a local path or HF repo
        is_local = os.path.exists(self.model_path) or self.model_path.startswith('/')
        load_kwargs = {}
        if is_local:
            load_kwargs['local_files_only'] = True
            logger.info("Using local files only (detected local path)")
        
        # Load processor
        self._processor = VibeVoiceProcessor.from_pretrained(
            self.model_path,
            **load_kwargs
        )
        
        # Load model with device-specific logic
        try:
            if self.device == "mps":
                self._model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=self.load_dtype,
                    attn_implementation=self.attn_impl,
                    device_map=None,
                    **load_kwargs
                )
                self._model.to("mps")
            elif self.device == "cuda":
                self._model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=self.load_dtype,
                    device_map="cuda",
                    attn_implementation=self.attn_impl,
                    **load_kwargs
                )
            else:  # cpu
                self._model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=self.load_dtype,
                    device_map="cpu",
                    attn_implementation=self.attn_impl,
                    **load_kwargs
                )
        except Exception as e:
            if self.attn_impl == 'flash_attention_2':
                logger.warning(f"Error with flash_attention_2: {e}")
                logger.info("Falling back to SDPA attention...")
                self._model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=self.load_dtype,
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                    attn_implementation='sdpa'
                )
                if self.device == "mps":
                    self._model.to("mps")
            else:
                raise e
        
        # Load checkpoint if provided
        if self.checkpoint_path:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}...")
            load_lora_assets(self._model, self.checkpoint_path)
        
        self._model.eval()
        self._model.set_ddpm_inference_steps(num_steps=self.num_inference_steps)
        
        logger.info("Model loaded successfully!")
    
    async def send_text(self, text: Optional[str]) -> None:
        """
        Queue text for generation with smart sentence buffering
        
        Args:
            text: Text to synthesize. None or empty string signals end of input.
        """
        if text is None:
            return
        
        if not text.strip():
            return
        
        # Just queue the text - sentence detection happens in the worker
        await self._input_queue.put(text)
    
    async def flush_buffer(self):
        """Force process remaining text (e.g., at end of stream)"""
        if self._text_buffer.strip():
            await self._input_queue.put(("FLUSH", self._text_buffer.strip()))
            self._text_buffer = ""
    
    async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
        """
        Receive audio chunk events as they're generated
        
        Yields:
            TTSChunkEvent: Audio chunks as they become available
        """
        # Start generation task if not already running
        if self._processing_task is None:
            self._processing_task = asyncio.create_task(self._generation_worker())
        
        try:
            while True:
                chunk = await self._output_queue.get()
                
                if chunk is None:
                    break
                
                if isinstance(chunk, Exception):
                    logger.error(f"Generation Error: {chunk}")
                    continue
                
                yield TTSChunkEvent.create(chunk)
                self._output_queue.task_done()
                
        except asyncio.CancelledError:
            pass
    
    async def _generation_worker(self):
        """Main generation loop that processes queued text"""
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        while not self._stop_event.is_set():
            try:
                text = await self._input_queue.get()
                
                if text is None:
                    break
                
                # Handle flush command
                if isinstance(text, tuple) and text[0] == "FLUSH":
                    text = text[1]
                
                # Process valid text
                if text.strip():
                    await loop.run_in_executor(
                        executor,
                        self._run_sync_stream,
                        text,
                        loop
                    )
                
                self._input_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in generation worker: {e}")
                loop.call_soon_threadsafe(
                    self._output_queue.put_nowait,
                    e
                )
        
        await self._output_queue.put(None)
        executor.shutdown(wait=False)
    
    def _run_sync_stream(self, text: str, loop: asyncio.AbstractEventLoop):
        """
        Synchronous generation with streaming output
        
        Args:
            text: Text to synthesize
            loop: Event loop for thread-safe callback
        """
        try:
            # Format text with speaker label (required by VibeVoice processor)
            formatted_text = self._format_text_with_speaker(text)
            
            # Prepare voice samples
            voice_samples = [self.voice_sample_path] if self.voice_sample_path else []
            
            inputs = self._processor(
                text=[formatted_text],
                voice_samples=[voice_samples] if voice_samples else None,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Move to device
            target_device = self.device if self.device != "cpu" else "cpu"
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(target_device)
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=self.cfg_scale,
                    tokenizer=self._processor.tokenizer,
                    generation_config={'do_sample': False},
                    verbose=False,
                    is_prefill=not self.disable_prefill,
                )
            
            # Extract and stream audio in chunks
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                audio = outputs.speech_outputs[0]
                
                # Convert to numpy
                if torch.is_tensor(audio):
                    # Convert bfloat16 to float32 first if needed
                    if audio.dtype == torch.bfloat16:
                        audio = audio.float()
                    audio = audio.cpu().numpy()
                
                # Flatten audio
                audio = audio.flatten()
                
                # Stream in chunks
                total_samples = len(audio)
                for i in range(0, total_samples, self.chunk_size):
                    if self._stop_event.is_set():
                        break
                    
                    chunk = audio[i:i + self.chunk_size]
                    
                    # Convert to PCM16 bytes
                    chunk_int16 = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
                    pcm_bytes = chunk_int16.tobytes()
                    
                    # Send to output queue thread-safely
                    loop.call_soon_threadsafe(
                        self._output_queue.put_nowait,
                        pcm_bytes
                    )
            
        except Exception as e:
            logger.error(f"Error in sync stream: {e}")
            loop.call_soon_threadsafe(
                self._output_queue.put_nowait,
                e
            )
    
    async def close(self) -> None:
        """Clean up resources"""
        # Process any remaining text in buffer before closing
        await self.flush_buffer()
        
        self._stop_event.set()
        await self._input_queue.put(None)
        
        if self._processing_task:
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        # Clear model from memory
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._processor is not None:
            del self._processor
            self._processor = None
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("VibeVoice TTS closed")
    
    def _format_text_with_speaker(self, text: str) -> str:
        """
        Format text with speaker labels, handling various input formats
        
        Args:
            text: Input text (may or may not have speaker labels)
            
        Returns:
            Text formatted with "Speaker X:" labels for each line/sentence
        """
        text = text.strip()
        if not text:
            return ""
        
        # If already formatted with Speaker labels, return as-is
        if text.startswith("Speaker"):
            return text
        
        # Split by newlines first
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line already has Speaker label
            if line.startswith("Speaker"):
                formatted_lines.append(line)
            else:
                # Split long lines by sentence endings for better prosody
                # This handles cases where LLM outputs multiple sentences
                sentences = self._split_into_sentences(line)
                for sentence in sentences:
                    if sentence.strip():
                        formatted_lines.append(f"Speaker 1: {sentence.strip()}")
        
        return '\n'.join(formatted_lines)
    
    def _split_into_sentences(self, text: str) -> list:
        """
        Split text into sentences for better prosody and streaming
        Handles English, Chinese, Malay text
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        import re
        
        # Don't split if text is very short
        if len(text) < 50:
            return [text]
        
        sentences = []
        
        # Pattern for sentence endings
        # English: . ! ? followed by space or end
        # Chinese: 。！？ (can be followed by anything)
        # Also split on \n
        sentence_pattern = r'([.!?。！？]+(?:\s+|$)|[\n]+)'
        
        # Split and keep delimiters
        parts = re.split(sentence_pattern, text)
        
        current_sentence = ""
        
        for part in parts:
            if not part.strip():
                continue
            
            # Check if this is a sentence delimiter
            if re.match(sentence_pattern, part):
                current_sentence += part.rstrip()
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part
        
        # Add remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # If no sentences found, split by length for Chinese/long text
        if not sentences or len(sentences) == 1 and len(sentences[0]) > 150:
            return self._split_by_length(text, max_length=100)
        
        return sentences if sentences else [text]
    
    def _split_by_length(self, text: str, max_length: int = 100) -> list:
        """
        Split text by length, trying to break at natural points
        Useful for Chinese text or text without clear sentence boundaries
        
        Args:
            text: Input text
            max_length: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        import re
        
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        
        # Try to break at Chinese commas (，、) or English commas/semicolons
        # Also break at conjunctions in Chinese
        break_points = r'[，、,;：: ]+'
        
        # Split into potential break points
        parts = re.split(f'({break_points})', text)
        
        current_chunk = ""
        
        for part in parts:
            # If adding this part exceeds max_length
            if len(current_chunk) + len(part) > max_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = part if not re.match(break_points, part) else ""
            else:
                current_chunk += part
        
        # Add remaining
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Final safety: if chunks are still too long, force split
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_length * 1.5:
                # Force split at max_length
                for i in range(0, len(chunk), max_length):
                    final_chunks.append(chunk[i:i + max_length])
            else:
                final_chunks.append(chunk)
        
        return final_chunks if final_chunks else [text]