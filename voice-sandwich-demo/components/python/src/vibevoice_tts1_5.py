import os
import threading
import traceback
import copy
import logging
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, List
import soundfile as sf  # <--- ADD THIS IMPORT AT THE TOP

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VibeVoiceStreaming1.5B")

BASE = Path(__file__).parent
SAMPLE_RATE = 24_000

class StreamingTTS15B:
    def __init__(
        self,
        model_path: str,
        voices_dir: str,  # Directory containing .wav files
        device: str = "cuda",
        inference_steps: int = 30,
    ) -> None:
        self.model_path = model_path
        self.voices_dir = voices_dir
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE
        
        # --- Device Setup ---
        if device == "mpx": device = "mps"
        if device == "mps" and not torch.backends.mps.is_available(): device = "cpu"
        self.device = device
        self._torch_device = torch.device(device)

        # --- Load Model & Processor ---
        self.load_model()
        
        # --- Load Voices ---
        # Maps "SpeakerName" -> Path_to_WAV
        self.voice_presets = self._scan_voice_files()
        
        # Cache for the computed tensors (so we don't re-process WAVs every time)
        self._voice_cache: Dict[str, Any] = {}
        
        # Set default voice
        self.default_voice_key = next(iter(self.voice_presets)) if self.voice_presets else None
        if self.default_voice_key:
            logger.info(f"Default voice set to: {self.default_voice_key}")
            # Pre-compute the default voice immediately
            self._ensure_voice_cached(self.default_voice_key)

    def load_model(self):
        logger.info(f"Loading 1.5B Processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        # --- 1.5B Specific Settings ---
        # 1.5B on GB10/Newer GPUs usually needs SDPA to avoid FlashAttn2 crashes
        # 1.5B uses bfloat16
        if self.device == "cuda":
            dtype = torch.bfloat16
            attn = "sdpa" # Enforce SDPA for stability
        elif self.device == "mps":
            dtype = torch.float32
            attn = "sdpa"
        else:
            dtype = torch.float32
            attn = "sdpa"

        logger.info(f"Loading 1.5B Model (dtype={dtype}, attn={attn})...")
        
        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=self.device,
                attn_implementation=attn,
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

        self.model.eval()
        
        # Setup Scheduler
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

    def _scan_voice_files(self) -> Dict[str, Path]:
        """Scans directory for .wav files only"""
        if not os.path.exists(self.voices_dir):
            logger.warning(f"Voices directory not found: {self.voices_dir}")
            return {}

        presets = {}
        # Find all .wav files
        for f in os.listdir(self.voices_dir):
            if f.lower().endswith(".wav"):
                # Clean name: "en-Alice_woman.wav" -> "Alice"
                name = os.path.splitext(f)[0]
                if '_' in name: name = name.split('_')[0]
                if '-' in name: name = name.split('-')[-1]
                
                full_path = Path(self.voices_dir) / f
                presets[name] = full_path
                presets[os.path.splitext(f)[0]] = full_path # Also add full filename as key

        logger.info(f"Found {len(presets)} voice samples (.wav)")
        return presets

    def _ensure_voice_cached(self, key: str) -> Any:
        if key not in self.voice_presets:
            logger.warning(f"Voice {key} not found, using default.")
            key = self.default_voice_key

        if key not in self._voice_cache:
            wav_path = self.voice_presets[key]
            logger.info(f"Computing Voice Cache for {key} from {wav_path}...")
            
            try:
                # --- FIX: Use soundfile instead of torchaudio.load ---
                # This bypasses the TorchCodec requirement entirely
                wav_numpy, sr = sf.read(str(wav_path))
                
                # soundfile returns numpy, we need torch tensor
                # soundfile shape is usually (samples, channels), we need (channels, samples)
                wav = torch.from_numpy(wav_numpy).float()
                
                # Handle Dimensions
                if wav.ndim == 1:
                    # Mono: (Samples) -> (1, Samples)
                    wav = wav.unsqueeze(0)
                else:
                    # Stereo/Multi: (Samples, Channels) -> (Channels, Samples)
                    wav = wav.t()
                
                # -----------------------------------------------------
                
                # 2. Resample to 24k if needed
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    wav = resampler(wav)
                
                # 3. Mono Mix (Ensure 1 channel)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)

                # 4. Process Prompt
                prompt_text = "This is a reference audio for voice cloning."
                
                inputs = self.processor.process_prompt(
                    audio=wav.squeeze().numpy(),
                    text=prompt_text,
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}

                # 5. Generate Cache
                with torch.no_grad():
                    outputs = self.model.model(
                        **inputs,
                        use_cache=True,
                        return_dict=True
                    )
                
                self._voice_cache[key] = outputs.past_key_values
                logger.info(f"Voice {key} cached successfully.")

            except Exception as e:
                logger.error(f"Error processing voice {key}: {e}")
                traceback.print_exc()
                return None

        return self._voice_cache[key]

    def _prepare_inputs(self, text: str, prefilled_outputs: object):
        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": prefilled_outputs,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }
        processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)
        return {k: v.to(self._torch_device) for k, v in processed.items()}

    def _run_generation(self, inputs, streamer, errors, cfg, temp, stop_event, prefilled_outputs):
        try:
            self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    "do_sample": False, # Usually False for 1.5B stability
                    "temperature": temp,
                },
                audio_streamer=streamer,
                stop_check_fn=stop_event.is_set,
                verbose=False,
                refresh_negative=True,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            )
        except Exception as e:
            errors.append(e)
            traceback.print_exc()
            streamer.end()

    def stream(
        self,
        text: str,
        voice_key: str = None,
        cfg_scale: float = 1.3,
        temperature: float = 0.3,
    ) -> Iterator[np.ndarray]:
        if not text.strip(): return

        # 1. Get the Voice Cache (Computed from WAV)
        voice_key = voice_key or self.default_voice_key
        prefilled_outputs = self._ensure_voice_cached(voice_key)
        
        if prefilled_outputs is None:
            logger.error("Failed to load voice resources.")
            return

        # 2. Prepare Inputs
        inputs = self._prepare_inputs(text, prefilled_outputs)
        
        # 3. Setup Streamer
        streamer = AudioStreamer(batch_size=1)
        stop_event = threading.Event()
        errors = []

        # 4. Run Thread
        thread = threading.Thread(
            target=self._run_generation,
            kwargs={
                "inputs": inputs,
                "streamer": streamer,
                "errors": errors,
                "cfg": cfg_scale,
                "temp": temperature,
                "stop_event": stop_event,
                "prefilled_outputs": prefilled_outputs
            },
            daemon=True
        )
        thread.start()

        # 5. Yield Audio
        try:
            for chunk in streamer.get_stream(0):
                if torch.is_tensor(chunk):
                    chunk = chunk.detach().cpu().float().numpy()
                else:
                    chunk = np.asarray(chunk, dtype=np.float32)
                
                # Normalize peak
                peak = np.max(np.abs(chunk))
                if peak > 1.0: chunk /= peak
                
                yield chunk
        finally:
            stop_event.set()
            streamer.end()
            thread.join()

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        return (chunk * 32767.0).astype(np.int16).tobytes()