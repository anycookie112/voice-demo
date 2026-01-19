# import asyncio
# from vibevoice_tts1_5 import StreamingTTS15B

# # 1. Initialize (Pointing to the 1.5B Model)
# tts_service = StreamingTTS15B(
#     model_path="/home/robust/models/VibeVoice-1.5B", # <--- 1.5B Model
#     voices_dir="/home/robust/voice_demo_docket/voice-demo/VibeVoice/demo/voices", # <--- Folder with .wav files
#     device="cuda",
#     inference_steps=30
# )

# # 2. Stream
# # This will auto-load "Alice.wav", compute the 128-size tensor, and stream audio
# text = "Hello, this is a test of the 1.5B model streaming from raw audio."
# stream = tts_service.stream(text, voice_key="Alice")

# for chunk in stream:
#     # Play or save audio
#     pass

"""
Test script for VibeVoice TTS
"""

import asyncio
import os
from vibevoice_new import VibeVoiceTTS


async def test_simple_generation():
    """Test simple text-to-speech generation"""
    print("Testing VibeVoice TTS...")
    
    # Initialize TTS
    tts = VibeVoiceTTS(
        model_path="microsoft/VibeVoice-1.5b",
        voice_sample_path="/home/robust/voice_demo_docket/voice-demo/VibeVoice/demo/voices/en-Alice_woman.wav",  # Update with your voice file path
        device=None,  # Auto-detect
        cfg_scale=1.3,
    )
    
    try:
        # Test 1: Simple generation
        print("\n--- Test 1: Simple generation ---")
        audio_bytes = await tts.generate_complete("Hello, this is a test of the VibeVoice text to speech system.")
        print(f"Generated audio: {len(audio_bytes)} bytes")
        
        # Save to file
        output_path = "test_output.wav"
        if len(audio_bytes) > 0:
            # Note: This saves raw PCM, you may need to add WAV header
            import wave
            import numpy as np
            
            # Convert bytes back to int16 array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Save as WAV
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)  # 24kHz
                wav_file.writeframes(audio_array.tobytes())
            
            print(f"Saved audio to {output_path}")
        
        # Test 2: Multiple generations
        print("\n--- Test 2: Multiple generations ---")
        texts = [
            "This is the first sentence.",
            "Here is another one.",
            "And finally, a third sentence."
        ]
        
        for i, text in enumerate(texts):
            print(f"Generating text {i+1}/{len(texts)}: {text}")
            audio = await tts.generate_complete(text)
            print(f"  Generated: {len(audio)} bytes")
        
    finally:
        # Always cleanup
        print("\nCleaning up...")
        await tts.close()
        print("Done!")


async def test_streaming():
    """Test streaming text-to-speech (placeholder for future implementation)"""
    print("\n--- Test 3: Streaming mode (not fully implemented) ---")
    
    tts = VibeVoiceTTS(
        model_path="microsoft/VibeVoice-1.5b",
        voice_sample_path="voices/andrew.wav",
    )
    
    try:
        # Send multiple text chunks
        await tts.send_text("Hello, ")
        await tts.send_text("this is ")
        await tts.send_text("streaming text.")
        await tts.send_text("")  # Signal end
        
        print("Text queued for streaming generation")
        
        # In a full implementation, you would:
        # async for chunk in tts.receive_events():
        #     print(f"Received chunk: {len(chunk.audio)} bytes")
        
    finally:
        await tts.close()


async def main():
    """Main test function"""
    print("="*60)
    print("VibeVoice TTS Test Suite")
    print("="*60)
    
    # Run test
    await test_simple_generation()
    
    # Uncomment to test streaming (not fully implemented)
    # await test_streaming()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())