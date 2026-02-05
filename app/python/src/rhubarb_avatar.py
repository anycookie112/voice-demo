# pip install rhubarb-lipsync-cli

import subprocess
import json

def generate_lipsync(audio_file_path: str, output_json_path: str):
    """Generate lip sync data using Rhubarb"""
    
    # Run Rhubarb CLI
    result = subprocess.run([
        'rhubarb',
        '-f', 'json',
        '-o', output_json_path,
        audio_file_path
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        with open(output_json_path, 'r') as f:
            return json.load(f)
    else:
        raise Exception(f"Rhubarb failed: {result.stderr}")

# In your TTS handler:
def text_to_speech_with_lipsync(text: str):
    # Generate audio with your TTS
    audio_file = generate_tts(text)  # Your existing TTS function
    
    # Generate lip sync data
    lipsync_data = generate_lipsync(audio_file, 'lipsync.json')
    
    return {
        'audio_url': '/audio/' + audio_file,
        'lipsync': lipsync_data
    }