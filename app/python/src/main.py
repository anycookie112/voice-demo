import re
import asyncio
import contextlib
import logging
import os
# CRITICAL: Import torch first to ensure bundled cuDNN is loaded before CTranslate2
try:
    import torch
except ImportError:
    pass

from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableGenerator
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("VoiceAgent")

from events import (
    AgentChunkEvent,
    AgentEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    VoiceAgentEvent,
    TTSStopEvent,
    LogEvent,
    event_to_dict,
)
from utils import merge_async_iters
from fasterwhisper_stt import LocalWhisperSTT 
from kokoro_tts import KokoroTTS
from agents import get_agent
# from vibevoice_tts import VibeVoiceAsyncTTS
# from vibevoice_new import VibeVoiceTTS

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


async def _stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Audio (Bytes) → Voice Events (VoiceAgentEvent)
    """
    logger.info("[System] Initializing STT Model (Whisper)...")
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
        base_silence_threshold=200.0,
        energy_window_size=5,
        speech_ratio_threshold=0.6,
        end_of_speech_silence=1.0,
        end_of_turn_silence=0.5,
        min_speech_duration=0.8,
        use_noise_reduction=False,  
        # device = "cpu",
        # compute_type = "int8",
    )
    logger.info("[System] STT Model Ready.")

    async def send_audio():
        """
        Background task that pumps audio chunks to the STT model.

        This runs concurrently with the main coroutine, continuously reading
        audio chunks from the input stream and forwarding them to the STT.
        When the input stream ends, it signals completion by closing the
        WebSocket connection.
        """
        try:
            # Stream each audio chunk to the STT as it arrives
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
        finally:
            # Signal that audio streaming is complete
            await stt.close()

    # Launch the audio sending task in the background
    # This allows us to simultaneously receive transcripts in the main coroutine
    send_task = asyncio.create_task(send_audio())

    try:
        # Consumer loop: receive and yield transcription events as they arrive
        # from the STT model. The receive_events() method yields transcripts
        # as they become available.
        async for event in stt.receive_events():
            yield event
    finally:
        # Cleanup: ensure the background task is cancelled and awaited
        with contextlib.suppress(asyncio.CancelledError):
            send_task.cancel()
            await send_task
        # Ensure the WebSocket connection is closed
        await stt.close()


def make_agent_stream(agent_executor):
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
                logger.debug(f"[1] STT Output received: {event.transcript}") 
                # Invoke LangChain Agent
                stream = agent_executor.astream(
                    {"messages": [HumanMessage(content=event.transcript)]},
                    {"configurable": {"thread_id": thread_id}},
                    stream_mode="messages",
                )

                logger.debug("[2] Starting agent stream...")
                full_response = ""  # Accumulate full response for debugging
                
                async for message, metadata in stream:
                    # --- PROCESS AI MESSAGES (TEXT) ---
                    if isinstance(message, AIMessage):
                        content = message.content
                        
                        # Handle different content types
                        if isinstance(content, list):
                            # Sometimes content is a list of dicts with 'text' keys
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            content = ''.join(text_parts)
                        
                        if isinstance(content, str) and content.strip():
                            full_response += content
                            logger.debug(f"[3] AIMessage chunk: '{content}'")
                            yield AgentChunkEvent.create(content)
                        
                        # --- PROCESS TOOL CALLS ---
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
                logger.debug(f"[4] Agent stream finished. Full response: '{full_response[:100]}...'")
                yield AgentEndEvent.create()

    return _agent_stream


async def _tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    
    logger.info("[System] Initializing TTS Model (Kokoro)...")
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
    tts = KokoroTTS() 
    logger.info("[System] TTS Model Ready.")
    
    # # vibe 1.5b
    # tts = VibeVoiceTTS(
    #     model_path="/home/robust/models/VibeVoice-1.5B",
    #     voice_sample_path="/app/voice-demo/VibeVoice/demo/voices/en-Alice_woman.wav",
    #     device="cuda",
    #     cfg_scale=1.3,
    #     chunk_size=2400,  # 0.1 seconds at 24kHz
    # )

    async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
        text_buffer = ""
        
        async for event in event_stream:
            yield event

            # handle interruption
            if event.type == "stt_chunk" or event.type == "stt_output":
                if event.transcript.strip():
                     if hasattr(tts, 'interrupt'):
                         tts.interrupt()
                     yield TTSStopEvent.create()
                     text_buffer = "" # clear buffer

            # handle language switching
            if event.type == "stt_output" and event.language:
                logger.info(f"[Main] Language detected: {event.language}")
                # Map Whisper language to Kokoro language
                # Whisper: 'en', 'zh', 'ms', 'yue', etc.
                # Kokoro: 'a'/'b' (English), 'z' (Chinese), 'j' (Japanese), etc.
                
                lang_map = {
                    'en': 'a', # Default to American English
                    'zh': 'z', # Chinese
                    # 'ms': 'a', # Malay -> English (Fallback/No specific Malay model yet?)
                    # Add more mappings as Kokoro supports them
                }
                
                target_lang = lang_map.get(event.language, 'a') # Default to English
                if hasattr(tts, 'set_language'):
                    tts.set_language(target_lang)

            # 2. Process Text for TTS
            if event.type == "agent_chunk":
                logger.debug(f"[TTS] Received agent_chunk: {event.text[:30]}...")
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
                            logger.debug(f"[TTS] Sending sentence to TTS: {sentence[:50]}...")
                            await tts.send_text(sentence)
                        
                        # Remove processed sentence from buffer
                        text_buffer = text_buffer[end_idx:]
                    else:
                        # No end of sentence found yet, keep buffering
                        break
            
            # 3. Flush remaining text when agent is done
            elif event.type == "agent_end":
                logger.debug(f"[TTS] Agent end, flushing buffer: {text_buffer[:30] if text_buffer else 'empty'}...")
                if text_buffer.strip():
                    await tts.send_text(text_buffer)
                text_buffer = "" # Reset for next turn

    try:
        # Merge the upstream (Agent) and downstream (TTS Audio) streams
        async for event in merge_async_iters(process_upstream(), tts.receive_events()):
            yield event
    finally:
        await tts.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, custom_prompt: str = None):
    await websocket.accept()
    
    if custom_prompt:
        msg = f"Custom System Prompt Injected: {custom_prompt[:50]}..."
        bar = "*" * 50
        logger.info(f"\n{bar}\n{msg}\n{bar}")
        await websocket.send_json(event_to_dict(LogEvent.create(msg)))
    else:
        msg = "Using Default System Prompt."
        logger.info(msg)
        await websocket.send_json(event_to_dict(LogEvent.create(msg)))

    current_agent = get_agent(system_prompt_override=custom_prompt)
    _agent_stream = make_agent_stream(current_agent)

    pipeline = (
        RunnableGenerator(_stt_stream)  # Audio -> STT events
        | RunnableGenerator(_agent_stream)  # STT events -> STT + Agent events
        | RunnableGenerator(_tts_stream)  # STT + Agent events -> All events
    )

    async def websocket_audio_stream() -> AsyncIterator[bytes]:
        """Async generator that yields audio bytes from the websocket."""
        try:
            while True:
                data = await websocket.receive_bytes()
                yield data
        except WebSocketDisconnect:
            logger.info("Client disconnected gracefully")
        except Exception as e:
            logger.warning(f"WebSocket receive error: {e}")

    try:
        output_stream = pipeline.atransform(websocket_audio_stream())

        # Process all events from the pipeline, sending events back to the client
        async for event in output_stream:
            try:
                await websocket.send_json(event_to_dict(event))
            except WebSocketDisconnect:
                logger.info("Client disconnected during send")
                break
            except Exception as e:
                logger.warning(f"Error sending event: {e}")
                break
    except WebSocketDisconnect:
        logger.info("Session ended by client")
    except asyncio.CancelledError:
        logger.info("Session cancelled")
    except Exception as e:
        logger.error(f"Session error: {e}")
    finally:
        logger.info("Session cleanup complete")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__": 
    # uvicorn.run("main:app", port=8015, reload=True)
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        # Point to where they were copied in the container
        # ssl_keyfile="/app/key.pem", 
        # ssl_certfile="/app/cert.pem"
    )
