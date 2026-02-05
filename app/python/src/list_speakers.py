
import torch
from qwen_tts import Qwen3TTSModel

model_path = "/home/robust/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"

try:
    model = Qwen3TTSModel.from_pretrained(model_path,
                                        device_map = 'auto',
                                        dtype = torch.bfloat16,
                                        attn_implementation = 'flash_attention_2')
    
    print("Supported speakers:")
    print(model.get_supported_speakers())
    
except Exception as e:
    print(f"Error: {e}")
