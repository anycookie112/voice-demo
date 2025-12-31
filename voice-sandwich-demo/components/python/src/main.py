import os
import asyncio
import contextlib
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableGenerator
from langgraph.checkpoint.memory import InMemorySaver
from starlette.staticfiles import StaticFiles
import re 

# from assemblyai_stt import AssemblyAISTT
# from components.python.src.cartesia_tts import CartesiaTTS
from events import (
    AgentChunkEvent,
    AgentEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    VoiceAgentEvent,
    event_to_dict,
)
from utils import merge_async_iters
from fasterwhisper_stt import LocalWhisperSTT 
from whisper_pytorch import WhisperPytorchSTT
from kokoro_tts import KokoroTTS
from models import get_ollama_model, get_groq_model
from vibevoice_tts import VibeVoiceAsyncTTS
from vibevoice_new import VibeVoiceTTS
load_dotenv()

# Static files are served from the shared web build output
STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

if not STATIC_DIR.exists():
    raise RuntimeError(
        f"Web build not found at {STATIC_DIR}. "
        "Run 'make build-web' or 'make dev-py' from the project root."
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def add_to_order(item: str, quantity: int) -> str:
    """Add an item to the customer's sandwich order."""
    return f"Added {quantity} x {item} to the order."


def confirm_order(order_summary: str) -> str:
    """Confirm the final order with the customer."""
    return f"Order confirmed: {order_summary}. Sending to kitchen."


system_prompt = """
You are a helpful sandwich shop assistant. Your goal is to take the user's order.
Be concise and friendly.

Available toppings: lettuce, tomato, onion, pickles, mayo, mustard.
Available meats: turkey, ham, roast beef.
Available cheeses: swiss, cheddar, provolone.

The price for any sandwich is $5 plus $1 for each topping, meat, or cheese.

${CARTESIA_TTS_SYSTEM_PROMPT}
"""
from cartesia_prompts import CARTESIA_TTS_SYSTEM_PROMPT
system_prompt_chatonly = """
You are a friendly voice assistant having a natural conversation with the user.
The user may speak in English, Malay, or Chinese, and you should respond in the same language or gently mix languages when it feels natural, like in real everyday speech.

Keep your responses concise, warm, and easy to listen to. Speak in a flowing, storytelling style, as if you are chatting with a friend rather than giving instructions or lists. Let your sentences connect smoothly, avoiding rigid structures or point-by-point explanations.

Do not use any markdown, symbols, or formatting. Output plain text only, suitable for a voice interface.
Your goal is to sound human, relaxed, and engaging, making the conversation feel natural and effortless.

${CARTESIA_TTS_SYSTEM_PROMPT}
"""
    


# 1. Check which provider to use (Defaults to "groq" if not set)
provider = os.getenv("LLM_PROVIDER", "groq").lower()

if provider == "ollama":
    print("--> Using LLM Provider: Ollama")
    llm = get_ollama_model()
else:
    print("--> Using LLM Provider: Groq")
    # 2. Get Key from Environment (Don't hardcode "gsk_...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables!")
    
    llm = get_groq_model(api_key=api_key)

from data_visualisation.main import main2 as make_agent
agent = create_agent(
    model=llm,
    tools=[add_to_order, confirm_order],
    system_prompt=system_prompt_chatonly,
    checkpointer=InMemorySaver(),
)


# agent = make_agent(llm)




async def _stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Audio (Bytes) → Voice Events (VoiceAgentEvent)

    This function takes a stream of audio chunks and sends them to AssemblyAI for STT.

    It uses a producer-consumer pattern where:
    - Producer: A background task reads audio chunks from audio_stream and sends
      them to AssemblyAI via WebSocket. This runs concurrently with the consumer,
      allowing transcription to begin before all audio has arrived.
    - Consumer: The main coroutine receives transcription events from AssemblyAI
      and yields them downstream. Events include both partial results (stt_chunk)
      and final transcripts (stt_output).

    Args:
        audio_stream: Async iterator of PCM audio bytes (16-bit, mono, 16kHz)

    Yields:
        STT events (stt_chunk for partials, stt_output for final transcripts)
    """
    # stt = WhisperPytorchSTT(
    #         model_size="large-v3-turbo",
    #         sample_rate=16000,          # <= IMPORTANT: use the WAV's SR (likely 24000)
    #         device="cuda",           # or "cpu" if you want CPU
    #         compute_type="float16",  # safe
    #         silence_threshold=50.0,  # make VAD more permissive
    #         min_silence_chunks=3,    # detect utterance quickly
    #     )
    # stt = LocalWhisperSTT(
    #     model_size="large-v3-turbo", # or "distil-large-v3" for 3x speed
    #     device="cuda",         # FORCE CUDA
    #     compute_type="float16" # FORCE FLOAT16
    # )
    # NEW IMPROVED WHISPER STT
    stt = LocalWhisperSTT(
        base_silence_threshold=700.0,
        energy_window_size=5,
        speech_ratio_threshold=0.6,
        end_of_speech_silence=1.0,
        end_of_turn_silence=0.5,
        min_speech_duration=0.8,
        use_noise_reduction=False,  # Set True if very noisy
    )

    async def send_audio():
        """
        Background task that pumps audio chunks to AssemblyAI.

        This runs concurrently with the main coroutine, continuously reading
        audio chunks from the input stream and forwarding them to AssemblyAI.
        When the input stream ends, it signals completion by closing the
        WebSocket connection.
        """
        try:
            # Stream each audio chunk to AssemblyAI as it arrives
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
        finally:
            # Signal to AssemblyAI that audio streaming is complete
            await stt.close()

    # Launch the audio sending task in the background
    # This allows us to simultaneously receive transcripts in the main coroutine
    send_task = asyncio.create_task(send_audio())

    try:
        # Consumer loop: receive and yield transcription events as they arrive
        # from AssemblyAI. The receive_events() method listens on the WebSocket
        # for transcript events and yields them as they become available.
        async for event in stt.receive_events():
            yield event
    finally:
        # Cleanup: ensure the background task is cancelled and awaited
        with contextlib.suppress(asyncio.CancelledError):
            send_task.cancel()
            await send_task
        # Ensure the WebSocket connection is closed
        await stt.close()


async def _agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    FIXED: Uses message.content instead of message.text
    """
    thread_id = str(uuid4())

    async for event in event_stream:
        # 1. Pass through all events (User Input, STT, etc.)
        yield event

        if event.type == "stt_output":
            print(f"DEBUG: [1] STT Output received: {event.transcript}") 
            # Invoke LangChain Agent
            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )

            async for message, metadata in stream:
                # --- PROCESS AI MESSAGES (TEXT) ---
                if isinstance(message, AIMessage):
                    # FIX 1: Use .content, not .text
                    content = message.content
                    
                    # FIX 2: LangChain sometimes yields empty chunks or list-based content
                    if isinstance(content, str) and content:
                        yield AgentChunkEvent.create(content)
                    
                    # --- PROCESS TOOL CALLS ---
                    # Note: handling streaming tool calls can be tricky.
                    # This assumes message.tool_calls is populated fully or accumulatively.
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            yield ToolCallEvent.create(
                                id=tool_call.get("id", str(uuid4())),
                                name=tool_call.get("name", "unknown"),
                                args=tool_call.get("args", {}),
                            )

                # --- PROCESS TOOL RESULTS ---
                if isinstance(message, ToolMessage):
                    yield ToolResultEvent.create(
                        tool_call_id=getattr(message, "tool_call_id", ""),
                        name=getattr(message, "name", "unknown"),
                        result=str(message.content) if message.content else "",
                    )

            # Signal end of turn
            yield AgentEndEvent.create()



async def _tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    
    # Initialize your TTS (VibeVoice or Kokoro)
    # tts = VibeVoiceAsyncTTS(model_path="/app/models/VibeVoice-Realtime-0.5B")
    # tts = VibeVoiceAsyncTTS(
    # model_path="/app/models/VibeVoice-1.5B",
    # device="cuda",
    # voice_preset=None,    # Or specific voice ID
    # inference_steps=60,   # High quality
    # temperature=0.3,
    # # hf_repo_id="microsoft/VibeVoice-1.5B",
    # hf_repo_id="microsoft/VibeVoice-Realtime-0.5B",
# )

    # kokoro tts
    # tts = KokoroTTS() 

    # vibe 1.5b
    tts = VibeVoiceTTS(
        model_path="/home/robust/models/VibeVoice-1.5B",
        voice_sample_path="/app/voice-demo/VibeVoice/demo/voices/en-Alice_woman.wav",
        device="cuda",
        cfg_scale=1.3,
        chunk_size=2400,  # 0.1 seconds at 24kHz
    )

# The rest of your _tts_stream code should now work!

    async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
        # Buffer to accumulate partial text chunks
        text_buffer = ""
        
        async for event in event_stream:
            # 1. Pass ALL events to the UI immediately (So text bubbles appear)
            yield event

            # 2. Process Text for TTS
            if event.type == "agent_chunk":
                text_buffer += event.text
                
                # Check if we have a full sentence (ends in . ? ! followed by space or newline)
                # We split iteratively to handle multiple sentences in one chunk
                while True:
                    # Regex: Find punctuation [.?!] followed by whitespace or end of string
                    match = re.search(r'([.?!]+)(\s+|$)', text_buffer)
                    if match:
                        end_idx = match.end()
                        sentence = text_buffer[:end_idx]
                        
                        # Send the complete sentence to TTS
                        if sentence.strip():
                            await tts.send_text(sentence)
                        
                        # Remove processed sentence from buffer
                        text_buffer = text_buffer[end_idx:]
                    else:
                        # No end of sentence found yet, keep buffering
                        break
            
            # 3. Flush remaining text when agent is done
            elif event.type == "agent_end":
                if text_buffer.strip():
                    await tts.send_text(text_buffer)
                text_buffer = "" # Reset for next turn

    try:
        # Merge the upstream (Agent) and downstream (TTS Audio) streams
        async for event in merge_async_iters(process_upstream(), tts.receive_events()):
            yield event
    finally:
        await tts.close()


pipeline = (
    RunnableGenerator(_stt_stream)  # Audio -> STT events
    | RunnableGenerator(_agent_stream)  # STT events -> STT + Agent events
    | RunnableGenerator(_tts_stream)  # STT + Agent events -> All events
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def websocket_audio_stream() -> AsyncIterator[bytes]:
        """Async generator that yields audio bytes from the websocket."""
        while True:
            data = await websocket.receive_bytes()
            yield data

    output_stream = pipeline.atransform(websocket_audio_stream())

    # Process all events from the pipeline, sending events back to the client
    async for event in output_stream:
        await websocket.send_json(event_to_dict(event))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    # uvicorn.run("main:app", port=8015, reload=True)
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        # Point to where they were copied in the container
        ssl_keyfile="/app/key.pem", 
        ssl_certfile="/app/cert.pem"
    )