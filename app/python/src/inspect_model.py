
import torch
from qwen_tts import Qwen3TTSModel

model_path = "/home/robust/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"

try:
    model = Qwen3TTSModel.from_pretrained(model_path,
                                        device_map = 'auto',
                                        dtype = torch.bfloat16,
                                        attn_implementation = 'flash_attention_2')
    
    print("Available attributes/methods:")
    for attr in dir(model):
        if not attr.startswith("_"):
            print(attr)
            
except Exception as e:
    print(e)
