import datetime
import builtins
import asyncio
import json
import os
import threading
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, cast

from langdetect import detect, LangDetectException


import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

import copy

BASE = Path(__file__).parent
SAMPLE_RATE = 24_000


def get_timestamp():
    timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc
    ).astimezone(
        datetime.timezone(datetime.timedelta(hours=8))
    ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return timestamp

class StreamingTTSService:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        inference_steps: int = 60,
    ) -> None:
        # Keep model_path as string for HuggingFace repo IDs (Path() converts / to \ on Windows)
        self.model_path = model_path
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE

        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self._voice_cache: Dict[str, Tuple[object, Path, str]] = {}

        if device == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            device = "mps"        
        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            device = "cpu"
        self.device = device
        self._torch_device = torch.device(device)

    def load(self) -> None:
        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        
        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = 'cuda'
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = 'cpu'
            attn_impl_primary = "sdpa"
        print(f"Using device: {device_map}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        # Load model
        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl_primary,
            )
            
            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print("Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality.")
                
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=self.device,
                    attn_implementation='sdpa',
                )
                print("Load model with SDPA successfully ")
            else:
                raise e

        self.model.eval()

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        self.voice_presets = self._load_voice_presets()
        preset_name = os.environ.get("VOICE_PRESET")
        self.default_voice_key = self._determine_voice_key(preset_name)
        self._ensure_voice_cached(self.default_voice_key)

    def _load_voice_presets(self) -> Dict[str, Path]:
        voices_dir = BASE.parent / "demo" / "voices" / "streaming_model"
        if not voices_dir.exists():
            raise RuntimeError(f"Voices directory not found: {voices_dir}")

        presets: Dict[str, Path] = {}
        for pt_path in voices_dir.rglob("*.pt"):
            presets[pt_path.stem] = pt_path

        if not presets:
            raise RuntimeError(f"No voice preset (.pt) files found in {voices_dir}")

        print(f"[startup] Found {len(presets)} voice presets")
        return dict(sorted(presets.items()))

    def _determine_voice_key(self, name: Optional[str]) -> str:
        if name and name in self.voice_presets:
            return name

        default_key = "sp-Spk0_woman"
        if default_key in self.voice_presets:
            return default_key

        first_key = next(iter(self.voice_presets))
        print(f"[startup] Using fallback voice preset: {first_key}")
        return first_key

    def _ensure_voice_cached(self, key: str) -> Tuple[object, Path, str]:
        if key not in self.voice_presets:
            raise RuntimeError(f"Voice preset {key!r} not found")

        if key not in self._voice_cache:
            preset_path = self.voice_presets[key]
            print(f"[startup] Loading voice preset {key} from {preset_path}")
            print(f"[startup] Loading prefilled prompt from {preset_path}")
            prefilled_outputs = torch.load(
                preset_path,
                map_location=self._torch_device,
                weights_only=False,
            )
            self._voice_cache[key] = prefilled_outputs

        return self._voice_cache[key]

    def _get_voice_resources(self, requested_key: Optional[str]) -> Tuple[str, object, Path, str]:
        key = requested_key if requested_key and requested_key in self.voice_presets else self.default_voice_key
        if key is None:
            key = next(iter(self.voice_presets))
            self.default_voice_key = key

        prefilled_outputs = self._ensure_voice_cached(key)
        return key, prefilled_outputs

    def _prepare_inputs(self, text: str, prefilled_outputs: object):
        if not self.processor or not self.model:
            raise RuntimeError("StreamingTTSService not initialized")

        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": prefilled_outputs,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }

        processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)

        prepared = {
            key: value.to(self._torch_device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
        return prepared

    def _run_generation(
        self,
        inputs,
        audio_streamer: AudioStreamer,
        errors,
        cfg_scale: float,
        do_sample: bool,
        temperature: float,
        top_p: float,
        refresh_negative: bool,
        prefilled_outputs,
        stop_event: threading.Event,
    ) -> None:
        try:
            self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                    "top_p": top_p if do_sample else 1.0,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=stop_event.is_set,
                verbose=False,
                refresh_negative=refresh_negative,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            )
        except Exception as exc:  # pragma: no cover - diagnostic logging
            errors.append(exc)
            traceback.print_exc()
            audio_streamer.end()

    def stream(
        self,
        text: str,
        cfg_scale: float = 4.0,
        do_sample: bool = False,
        temperature: float = 0.3,
        top_p: float = 0.9,
        refresh_negative: bool = True,
        inference_steps: Optional[int] = None,
        voice_key: Optional[str] = None,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        # --- NEW LANGUAGE CHECK START ---
        if not self._is_allowed_language(text):
            print(f"Skipping generation: Text '{text[:20]}...' is not in Mandarin, English, or Malay.")
            return # Stop here, yield nothing

        text = text.replace("’", "'")
        selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)

        def emit(event: str, **payload: Any) -> None:
            if log_callback:
                try:
                    log_callback(event, **payload)
                except Exception as exc:
                    print(f"[log_callback] Error while emitting {event}: {exc}")

        steps_to_use = self.inference_steps
        if inference_steps is not None:
            try:
                parsed_steps = int(inference_steps)
                if parsed_steps > 0:
                    steps_to_use = parsed_steps
            except (TypeError, ValueError):
                pass
        if self.model:
            self.model.set_ddpm_inference_steps(num_steps=steps_to_use)
        self.inference_steps = steps_to_use

        inputs = self._prepare_inputs(text, prefilled_outputs)
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list = []
        stop_signal = stop_event or threading.Event()

        thread = threading.Thread(
            target=self._run_generation,
            kwargs={
                "inputs": inputs,
                "audio_streamer": audio_streamer,
                "errors": errors,
                "cfg_scale": cfg_scale,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "refresh_negative": refresh_negative,
                "prefilled_outputs": prefilled_outputs,
                "stop_event": stop_signal,
            },
            daemon=True,
        )
        thread.start()

        generated_samples = 0

        try:
            stream = audio_streamer.get_stream(0)
            for audio_chunk in stream:
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)

                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak

                generated_samples += int(audio_chunk.size)
                emit(
                    "model_progress",
                    generated_sec=generated_samples / self.sample_rate,
                    chunk_sec=audio_chunk.size / self.sample_rate,
                )

                chunk_to_yield = audio_chunk.astype(np.float32, copy=False)

                yield chunk_to_yield
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join()
            if errors:
                emit("generation_error", message=str(errors[0]))
                raise errors[0]

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk * 32767.0).astype(np.int16)
        return pcm.tobytes()


    def test(self):
        print("Running test method in StreamingTTSService")
        return "test successful"


    def _is_allowed_language(self, text: str) -> bool:
        """
        Returns True if text is likely Mandarin, English, or Malay.
        """
        # 1. Mandatory check for Mandarin (Chinese characters)
        # If the text contains CJK Unified Ideographs, assume it is Mandarin.
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            return True

        # 2. Use Library for Latin-script languages (English vs Malay vs Others)
        try:
            lang = detect(text)
            
            # 'en' = English
            # 'ms' = Malay
            # 'id' = Indonesian (Linguistically very similar to Malay, often detected as 'id')
            allowed_codes = {'en', 'ms', 'id'} 
            
            if lang in allowed_codes:
                return True
                
            print(f"[Language Filter] Blocked language detected: {lang}")
            return False

        except LangDetectException:
            # Text was too short or ambiguous (e.g., numbers "123" or symbols "?!")
            # Default behavior: Allow it, as it's likely just punctuation or short replies.
            return True

services = StreamingTTSService(model_path = "/home/robust/models/VibeVoice-Realtime-0.5B")
services.test()


# import datetime
# import builtins
# import asyncio
# import json
# import os
# import threading
# import traceback
# from pathlib import Path
# from queue import Empty, Queue
# from typing import Any, Callable, Dict, Iterator, Optional, Tuple, cast

# import numpy as np
# import torch
# import torchaudio # Required for processing .wav files
# import copy

# from vibevoice.modular.modeling_vibevoice_streaming_inference import (
#     VibeVoiceStreamingForConditionalGenerationInference,
# )
# from vibevoice.processor.vibevoice_streaming_processor import (
#     VibeVoiceStreamingProcessor,
# )
# from vibevoice.modular.streamer import AudioStreamer
# from langdetect import detect, LangDetectException

# BASE = Path(__file__).parent
# SAMPLE_RATE = 24_000

# class StreamingTTSService:
#     def __init__(
#         self,
#         model_path: str,
#         device: str = "cuda",
#         inference_steps: int = 60,
#     ) -> None:
#         self.model_path = model_path
        
#         # Detect Model Version for Settings
#         self.is_1_5b = "1.5" in model_path or "1.5b" in model_path.lower()
#         if self.is_1_5b:
#             print(f"[Config] 1.5B Model Detected. Using SDPA and High Quality settings.")
#         else:
#             print(f"[Config] 0.5B Model Detected. Using Standard settings.")

#         self.inference_steps = inference_steps
#         self.sample_rate = SAMPLE_RATE
#         self.processor: Optional[VibeVoiceStreamingProcessor] = None
#         self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
#         self.voice_presets: Dict[str, Path] = {}
#         self.default_voice_key: Optional[str] = None
#         self._voice_cache: Dict[str, Tuple[object, Path, str]] = {}

#         # Device Standardization
#         if device == "mpx": device = "mps"        
#         if device == "mps" and not torch.backends.mps.is_available(): device = "cpu"
#         self.device = device
#         self._torch_device = torch.device(device)

#     def load(self) -> None:
#         print(f"[startup] Loading processor from {self.model_path}")
#         self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

#         # Decide Attention Implementation
#         if self.device == "mps":
#             load_dtype = torch.float32
#             device_map = None
#             attn_impl_primary = "sdpa"
#         elif self.device == "cuda":
#             load_dtype = torch.bfloat16
#             device_map = 'cuda'
#             # Force SDPA for 1.5B model to prevent GB10 hardware crashes
#             attn_impl_primary = "sdpa" if self.is_1_5b else "flash_attention_2"
#         else:
#             load_dtype = torch.float32
#             device_map = 'cpu'
#             attn_impl_primary = "sdpa"

#         print(f"Using device: {device_map}, dtype: {load_dtype}, attention: {attn_impl_primary}")
        
#         try:
#             self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
#                 self.model_path,
#                 torch_dtype=load_dtype,
#                 device_map=device_map,
#                 attn_implementation=attn_impl_primary,
#             )
#             if self.device == "mps": self.model.to("mps")
#         except Exception as e:
#             if attn_impl_primary == 'flash_attention_2':
#                 print("Flash Attention failed. Falling back to SDPA.")
#                 self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
#                     self.model_path,
#                     torch_dtype=load_dtype,
#                     device_map=self.device,
#                     attn_implementation='sdpa',
#                 )
#             else:
#                 raise e

#         self.model.eval()
#         self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
#             self.model.model.noise_scheduler.config,
#             algorithm_type="sde-dpmsolver++",
#             beta_schedule="squaredcos_cap_v2",
#         )
#         self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

#         # Load voices (supports .pt and .wav)
#         self.voice_presets = self._load_voice_presets()
#         preset_name = os.environ.get("VOICE_PRESET")
#         self.default_voice_key = self._determine_voice_key(preset_name)
        
#         # Pre-cache the default voice immediately
#         print(f"[startup] Initializing default voice: {self.default_voice_key}")
#         self._ensure_voice_cached(self.default_voice_key)

#     def _load_voice_presets(self) -> Dict[str, Path]:
#         # Switch folder based on model version, but default to 'streaming_model' if 1.5b folder is missing
#         if self.is_1_5b:
#             # You can organize folders, or just dump everything in one. 
#             # Here we look in '1.5b' first, but if it doesn't exist, fallback to 'streaming_model'
#             # (Note: .wav files in 'streaming_model' will work fine for 1.5B!)
#             target_folder = BASE.parent / "demo" / "voices" / "1.5b"
#             if not target_folder.exists():
#                 target_folder = BASE.parent / "demo" / "voices" / "streaming_model"
#         else:
#             target_folder = BASE.parent / "demo" / "voices" / "streaming_model"

#         if not target_folder.exists():
#             print(f"Warning: Voice folder {target_folder} does not exist. Creating it.")
#             os.makedirs(target_folder, exist_ok=True)

#         presets: Dict[str, Path] = {}
        
#         # 1. Look for .pt files (Cached Tensors)
#         for f in target_folder.rglob("*.pt"):
#             presets[f.stem] = f
            
#         # 2. Look for .wav files (Raw Audio)
#         # Note: If a file exists as both .pt and .wav, .pt takes precedence in this dict update
#         # unless we flip the order. We let .pt overwrite .wav (faster load).
#         for f in target_folder.rglob("*.wav"):
#             if f.stem not in presets: # Prefer existing .pt if available
#                 presets[f.stem] = f

#         print(f"[startup] Found {len(presets)} voice presets (PT + WAV) in {target_folder}")
#         return dict(sorted(presets.items()))

#     def _determine_voice_key(self, name: Optional[str]) -> str:
#         if name and name in self.voice_presets:
#             return name
        
#         # Try finding any file with 'woman' or 'female' in it
#         for key in self.voice_presets:
#             if "woman" in key.lower() or "female" in key.lower():
#                 return key

#         if self.voice_presets:
#             return next(iter(self.voice_presets))
        
#         raise RuntimeError("No voice presets found! Please add .wav or .pt files to the voices folder.")

#     def _ensure_voice_cached(self, key: str) -> object:
#         """
#         Loads the voice memory. 
#         - If .pt: Loads directly.
#         - If .wav: Processes audio -> Generates Tensor -> Caches in memory.
#         """
#         if key not in self.voice_presets:
#             raise RuntimeError(f"Voice preset {key!r} not found")

#         if key not in self._voice_cache:
#             file_path = self.voice_presets[key]
#             print(f"[VoiceLoader] Loading {key} from {file_path}")
            
#             if file_path.suffix == ".pt":
#                 # Legacy Loading (Must match model size 64 vs 128)
#                 prefilled_outputs = torch.load(
#                     file_path,
#                     map_location=self._torch_device,
#                     weights_only=False,
#                 )
#             elif file_path.suffix == ".wav":
#                 # Dynamic Generation (Compatible with ALL models)
#                 prefilled_outputs = self._generate_tensor_from_wav(file_path)
#             else:
#                 raise ValueError(f"Unsupported file type: {file_path}")

#             self._voice_cache[key] = prefilled_outputs

#         return self._voice_cache[key]

#     def _generate_tensor_from_wav(self, wav_path: Path):
#         """
#         Reads a .wav file and computes the KV Cache using the currently loaded model.
#         This fixes the 0.5B vs 1.5B compatibility issue automatically.
#         """
#         print(f"[VoiceLoader] Processing WAV file: {wav_path}")
#         # 1. Load Audio
#         wav, sr = torchaudio.load(str(wav_path))
#         # 2. Resample if necessary
#         if sr != self.sample_rate:
#             resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
#             wav = resampler(wav)
#         # 3. Mix to Mono
#         if wav.shape[0] > 1:
#             wav = wav.mean(dim=0, keepdim=True)
#         # 4. Dummy text prompt (Use generic text for voice cloning context)
#         # Using a standard prompt helps stabilize the voice style.
#         text_prompt = "This is a voice cloning reference audio."
#         # 5. Process Inputs
#         # Convert to numpy for processor
#         wav_numpy = wav.squeeze().numpy()
#         inputs = self.processor.process_prompt(
#             audio=wav_numpy,
#             text=text_prompt,
#             return_tensors="pt"
#         )
#         # Move to GPU
#         inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
#         # 6. Run Forward Pass to get Cache
#         with torch.no_grad():
#             outputs = self.model.model(
#                 **inputs,
#                 use_cache=True,
#                 return_dict=True
#             )
#         print(f"[VoiceLoader] Successfully generated voice cache for {wav_path.name}")
#         return outputs.past_key_values

#     def _get_voice_resources(self, requested_key: Optional[str]) -> Tuple[str, object, Path, str]:
#         key = requested_key if requested_key and requested_key in self.voice_presets else self.default_voice_key
#         if key is None:
#             key = next(iter(self.voice_presets))
#             self.default_voice_key = key
#         prefilled_outputs = self._ensure_voice_cached(key)
#         return key, prefilled_outputs

#     def _prepare_inputs(self, text: str, prefilled_outputs: object):
#         if not self.processor or not self.model:
#             raise RuntimeError("StreamingTTSService not initialized")
#         processor_kwargs = {
#             "text": text.strip(),
#             "cached_prompt": prefilled_outputs,
#             "padding": True,
#             "return_tensors": "pt",
#             "return_attention_mask": True,
#         }
#         processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)
#         prepared = {
#             key: value.to(self._torch_device) if hasattr(value, "to") else value
#             for key, value in processed.items()
#         }
#         return prepared

#     def _run_generation(
#         self,
#         inputs,
#         audio_streamer: AudioStreamer,
#         errors,
#         cfg_scale: float,
#         do_sample: bool,
#         temperature: float,
#         top_p: float,
#         refresh_negative: bool,
#         prefilled_outputs,
#         stop_event: threading.Event,
#     ) -> None:
#         try:
#             self.model.generate(
#                 **inputs,
#                 max_new_tokens=None,
#                 cfg_scale=cfg_scale,
#                 tokenizer=self.processor.tokenizer,
#                 generation_config={
#                     "do_sample": do_sample,
#                     "temperature": temperature if do_sample else 1.0,
#                     "top_p": top_p if do_sample else 1.0,
#                 },
#                 audio_streamer=audio_streamer,
#                 stop_check_fn=stop_event.is_set,
#                 verbose=False,
#                 refresh_negative=refresh_negative,
#                 all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
#             )
#         except Exception as exc:  # pragma: no cover - diagnostic logging
#             errors.append(exc)
#             traceback.print_exc()
#             audio_streamer.end()

#     def stream(
#         self,
#         text: str,
#         cfg_scale: float = 4.0,
#         do_sample: bool = False,
#         temperature: float = 0.3,
#         top_p: float = 0.9,
#         refresh_negative: bool = True,
#         inference_steps: Optional[int] = None,
#         voice_key: Optional[str] = None,
#         log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
#         stop_event: Optional[threading.Event] = None,
#     ) -> Iterator[np.ndarray]:
#         if not text.strip():
#             return
#         # --- NEW LANGUAGE CHECK START ---
#         if not self._is_allowed_language(text):
#             print(f"Skipping generation: Text '{text[:20]}...' is not in Mandarin, English, or Malay.")
#             return # Stop here, yield nothing
#         text = text.replace("’", "'")
#         selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)
#         def emit(event: str, **payload: Any) -> None:
#             if log_callback:
#                 try:
#                     log_callback(event, **payload)
#                 except Exception as exc:
#                     print(f"[log_callback] Error while emitting {event}: {exc}")
#         steps_to_use = self.inference_steps
#         if inference_steps is not None:
#             try:
#                 parsed_steps = int(inference_steps)
#                 if parsed_steps > 0:
#                     steps_to_use = parsed_steps
#             except (TypeError, ValueError):
#                 pass
#         if self.model:
#             self.model.set_ddpm_inference_steps(num_steps=steps_to_use)
#         self.inference_steps = steps_to_use
#         inputs = self._prepare_inputs(text, prefilled_outputs)
#         audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
#         errors: list = []
#         stop_signal = stop_event or threading.Event()
#         thread = threading.Thread(
#             target=self._run_generation,
#             kwargs={
#                 "inputs": inputs,
#                 "audio_streamer": audio_streamer,
#                 "errors": errors,
#                 "cfg_scale": cfg_scale,
#                 "do_sample": do_sample,
#                 "temperature": temperature,
#                 "top_p": top_p,
#                 "refresh_negative": refresh_negative,
#                 "prefilled_outputs": prefilled_outputs,
#                 "stop_event": stop_signal,
#             },
#             daemon=True,
#         )
#         thread.start()
#         generated_samples = 0
#         try:
#             stream = audio_streamer.get_stream(0)
#             for audio_chunk in stream:
#                 if torch.is_tensor(audio_chunk):
#                     audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
#                 else:
#                     audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
#                 if audio_chunk.ndim > 1:
#                     audio_chunk = audio_chunk.reshape(-1)
#                 peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
#                 if peak > 1.0:
#                     audio_chunk = audio_chunk / peak
#                 generated_samples += int(audio_chunk.size)
#                 emit(
#                     "model_progress",
#                     generated_sec=generated_samples / self.sample_rate,
#                     chunk_sec=audio_chunk.size / self.sample_rate,
#                 )
#                 chunk_to_yield = audio_chunk.astype(np.float32, copy=False)
#                 yield chunk_to_yield
#         finally:
#             stop_signal.set()
#             audio_streamer.end()
#             thread.join()
#             if errors:
#                 emit("generation_error", message=str(errors[0]))
#                 raise errors[0]

#     def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
#         chunk = np.clip(chunk, -1.0, 1.0)
#         pcm = (chunk * 32767.0).astype(np.int16)
#         return pcm.tobytes()

#     def test(self):
#         print("Running test method in StreamingTTSService")
#         return "test successful"

#     def _is_allowed_language(self, text: str) -> bool:
#         """
#         Returns True if text is likely Mandarin, English, or Malay.
#         """
#         # 1. Mandatory check for Mandarin (Chinese characters)
#         # If the text contains CJK Unified Ideographs, assume it is Mandarin.
#         if any("\u4e00" <= char <= "\u9fff" for char in text):
#             return True
#         # 2. Use Library for Latin-script languages (English vs Malay vs Others)
#         try:
#             lang = detect(text)
#             # 'en' = English
#             # 'ms' = Malay
#             # 'id' = Indonesian (Linguistically very similar to Malay, often detected as 'id')
#             allowed_codes = {'en', 'ms', 'id'} 
#             if lang in allowed_codes:
#                 return True
#             print(f"[Language Filter] Blocked language detected: {lang}")
#             return False
#         except LangDetectException:
#             # Text was too short or ambiguous (e.g., numbers "123" or symbols "?!")
#             # Default behavior: Allow it, as it's likely just punctuation or short replies.
#             return True
