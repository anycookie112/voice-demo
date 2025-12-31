import torch
import torchaudio
import argparse
import os
from pathlib import Path
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)

# --- CONFIGURATION ---
MODEL_PATH = "/home/robust/models/VibeVoice-1.5B" # <--- POINT TO YOUR 1.5B MODEL
DEVICE = "cuda"
SAMPLE_RATE = 24000
# ---------------------

def create_preset(audio_path, text, output_path):
    print(f"Loading 1.5B Model from: {MODEL_PATH}")
    
    # 1. Load Processor & Model
    processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_PATH)
    
    # Note: 1.5B usually uses bfloat16
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="sdpa", # Use SDPA for compatibility
    )
    model.eval()

    # 2. Load and Resample Audio
    print(f"Processing Audio: {audio_path}")
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        wav = resampler(wav)
    
    # Mix to mono if stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # 3. Process the Prompt
    # We feed the audio + the text transcription of that audio into the model
    # to generate the "Memory" (KV Cache)
    inputs = processor.process_prompt(
        audio=wav.squeeze().numpy(),
        text=text,
        return_tensors="pt"
    )
    
    # Move inputs to GPU
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 4. Generate the Cache (The ".pt" file content)
    print("Generating Voice Cache (Forward Pass)...")
    with torch.no_grad():
        # We run a forward pass to get the 'past_key_values'
        outputs = model.model(
            **inputs,
            use_cache=True,
            return_dict=True
        )
        # Extract the KV cache
        past_key_values = outputs.past_key_values

    # 5. Save the Preset
    print(f"Saving 1.5B compatible preset to: {output_path}")
    torch.save(past_key_values, output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to a short .wav file (3-10s) of the voice")
    parser.add_argument("--text", type=str, required=True, help="The exact text spoken in the .wav file")
    parser.add_argument("--output", type=str, default="voice_1.5b.pt", help="Where to save the .pt file")
    args = parser.parse_args()

    create_preset(args.audio, args.text, args.output)

"""
python3 converter.py \
  --audio /home/robust/voice_demo_docket/voice-demo/VibeVoice/demo/voices/en-Alice_woman.wav \
  --text "Hello this is a test of the voice cloning system." \
  --output /home/robust/voice_demo_docket/voice-demo/VibeVoice/demo/outputs/my_new_voice_1.5b.pt

    # arm codec support 
  uv pip install --prerelease=allow torchcodec

"""