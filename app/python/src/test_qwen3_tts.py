"""Simple test for Qwen3TTS class"""

import asyncio
import torch
from qwen3_tts import Qwen3TTS


async def test_qwen3_tts():
    """Test basic TTS generation"""
    
    # Initialize the TTS model
    model_path = "/home/robust/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    
    print("Initializing Qwen3TTS...")
    tts = Qwen3TTS(
        model_path=model_path,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        language="English",
        speaker="Ryan",
        instruct=None,
        sample_rate=24000,
        chunk_ms=50,
    )
    
    print("TTS initialized successfully!\n")
    
    # Send test text
    test_text = "Hello, this is a test of the Qwen3 text to speech system."
    print(f"Sending text: '{test_text}'")
    await tts.send_text(test_text)
    
    # Collect audio chunks
    print("Receiving audio chunks...")
    chunk_count = 0
    total_bytes = 0
    
    async for event in tts.receive_events():
        chunk_count += 1
        total_bytes += len(event.audio)
        if chunk_count <= 5 or chunk_count % 10 == 0:
            print(f"  Received chunk {chunk_count}: {len(event.audio)} bytes")
    
    print(f"\nTest completed!")
    print(f"Total chunks: {chunk_count}")
    print(f"Total audio bytes: {total_bytes}")
    print(f"Duration: ~{total_bytes / (24000 * 2):.2f} seconds")
    
    # Clean up
    await tts.close()
    print("TTS closed successfully!")


async def test_wav_output():
    """Test WAV file output"""
    
    model_path = "/home/robust/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    
    print("\nTesting WAV file output...")
    tts = Qwen3TTS(
        model_path=model_path,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        language="Chinese",
        speaker="Vivian",
        instruct="用特别夹子音又性感的语气说",
    )
    
    text = "你好，这是一个测试。"
    output_path = "test_qwen3_output.wav"
    
    await tts.synthesize_to_wav(text, output_path)
    print(f"Audio saved to {output_path}")
    
    await tts.close()


if __name__ == "__main__":
    print("=== Qwen3TTS Test Suite ===\n")
    
    # Run streaming test
    asyncio.run(test_qwen3_tts())
    
    # Run WAV file test
    asyncio.run(test_wav_output())
    
    print("\n=== All tests completed! ===")
