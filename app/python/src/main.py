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
    MarkdownChunkEvent,
    TTSTextEvent,
    TTSInstructEvent,
    event_to_dict,
)
from utils import merge_async_iters
from fasterwhisper_stt import LocalWhisperSTT, clear_whisper_model_cache 
from kokoro_tts import KokoroTTS
from agents import get_agent
from response_parser import parse_agent_response
from qwen3_tts import Qwen3TTS
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




def make_agent_stream(agent_executor):
    async def _agent_stream(
        event_stream: AsyncIterator[VoiceAgentEvent],
    ) -> AsyncIterator[VoiceAgentEvent]:
        """
        Transform stream: STT events → Agent events
        Streams <MARKDOWN>, <INSTRUCT>, and <TTS> content in real-time.
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
                
                # Streaming Parser State
                buffer = ""
                active_tag: str | None = None  # "MARKDOWN", "INSTRUCT", or "TTS"

                async for message, metadata in stream:
                    # --- PROCESS AI MESSAGES (TEXT) ---
                    if isinstance(message, AIMessage):
                        content = message.content
                        
                        # Handle different content types
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            content = ''.join(text_parts)
                        
                        if isinstance(content, str) and content:
                            buffer += content
                            
                            # State Machine for Tag Parsing
                            while True:
                                if active_tag is None:
                                    # Look for new tags
                                    m_start = buffer.find("<MARKDOWN>")
                                    i_start = buffer.find("<INSTRUCT>")
                                    t_start = buffer.find("<TTS>")
                                    
                                    # Find the earliest tag
                                    starts = []
                                    if m_start != -1:
                                        starts.append((m_start, "MARKDOWN", 10))
                                    if i_start != -1:
                                        starts.append((i_start, "INSTRUCT", 10))
                                    if t_start != -1:
                                        starts.append((t_start, "TTS", 5))
                                    
                                    if starts:
                                        # Sort by position to find earliest
                                        starts.sort(key=lambda x: x[0])
                                        pos, tag_name, tag_len = starts[0]
                                        buffer = buffer[pos + tag_len:]
                                        active_tag = tag_name
                                        continue
                                    else:
                                        # No start tag found. Keep buffer for next chunk.
                                        break
                                
                                else:
                                    # Inside a tag: Look for closing tag
                                    end_tag = f"</{active_tag}>"
                                    end_idx = buffer.find(end_tag)
                                    
                                    if end_idx != -1:
                                        # Found closing tag
                                        chunk = buffer[:end_idx]
                                        buffer = buffer[end_idx + len(end_tag):]
                                        
                                        if chunk:
                                            if active_tag == "MARKDOWN":
                                                yield MarkdownChunkEvent.create(chunk)
                                            elif active_tag == "INSTRUCT":
                                                # INSTRUCT is emitted as a single event (not streamed)
                                                yield TTSInstructEvent.create(chunk.strip())
                                            else:  # TTS
                                                yield TTSTextEvent.create(chunk)
                                        
                                        active_tag = None
                                        continue # Re-evaluate buffer for next tag
                                    
                                    else:
                                        # No closing tag yet. Yield partial content safely.
                                        # For INSTRUCT, we wait for the full content (don't stream)
                                        if active_tag == "INSTRUCT":
                                            break  # Wait for closing tag
                                        
                                        # We must NOT yield partial tags (e.g. "</MARK")
                                        # Look for the start of a potential closing tag ("<")
                                        last_open = buffer.rfind("<")
                                        
                                        if last_open != -1:
                                            # Yield up to the last "<"
                                            to_yield = buffer[:last_open]
                                            buffer = buffer[last_open:]
                                        else:
                                            # Safe to yield everything
                                            to_yield = buffer
                                            buffer = ""
                                            
                                        if to_yield:
                                            if active_tag == "MARKDOWN":
                                                yield MarkdownChunkEvent.create(to_yield)
                                            else:  # TTS
                                                yield TTSTextEvent.create(to_yield)
                                        break # Need more data

                        
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

                # Flush any remaining content in buffer after stream ends
                if buffer and active_tag:
                     logger.warning(f"Stream ended with unclosed {active_tag} tag. Flushing buffer.")
                     if active_tag == "MARKDOWN":
                         yield MarkdownChunkEvent.create(buffer)
                     else:
                         yield TTSTextEvent.create(buffer)

                logger.debug("[4] Agent stream finished.")
                
                # Signal end of turn
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
    # tts = KokoroTTS() 
    tts = Qwen3TTS(
        model_path="/home/robust/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        language="auto",  # lowercase required: auto, chinese, english, etc.
        speaker="Vivian",
        instruct="Use a professional tone"
    )

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

            # handle interruption - only on stt_output (final transcription)
            # stt_chunk is just preview, don't interrupt TTS for that
            if event.type == "stt_output":
                if event.transcript.strip():
                     if hasattr(tts, 'interrupt'):
                         tts.interrupt()
                     yield TTSStopEvent.create()
                     text_buffer = "" # clear buffer

            # handle language switching (only for Kokoro TTS, Qwen3TTS uses auto)
            if event.type == "stt_output" and event.language:
                logger.info(f"[Main] Language detected: {event.language}")
                # Qwen3TTS uses language="auto" and handles detection internally
                # Only apply mapping for Kokoro TTS
                if hasattr(tts, 'set_language'):
                    # Kokoro: 'a'/'b' (English), 'z' (Chinese), 'j' (Japanese), etc.
                    lang_map = {
                        'en': 'a',
                        'zh': 'z',
                        'yue': 'z',
                        'ms': 'a',
                    }
                    target_lang = lang_map.get(event.language, 'a')
                    tts.set_language(target_lang)

            # Handle TTS Instruct event (dynamic voice tone control for Qwen3TTS)
            if event.type == "tts_instruct":
                logger.info(f"[TTS] Voice instruct received: {event.instruct[:50]}...")
                if hasattr(tts, 'set_instruct'):
                    tts.set_instruct(event.instruct)

            # NEW: Process TTSTextEvent (parsed TTS content from <TTS> tags)
            if event.type == "tts_text":
                logger.debug(f"[TTS] Received tts_text: {event.text[:50]}...")
                text_buffer += event.text
                
                # Check if we have a full sentence
                # Include both English (.?!) and Chinese (。？！) punctuation
                while True:
                    match = re.search(r'([.?!。？！]+)(\s*|$)', text_buffer)
                    if match:
                        end_idx = match.end()
                        sentence = text_buffer[:end_idx]
                        
                        if sentence.strip():
                            logger.debug(f"[TTS] Sending sentence to TTS: {sentence[:50]}...")
                            await tts.send_text(sentence)
                        
                        text_buffer = text_buffer[end_idx:]
                    else:
                        break
            
            # Flush remaining text when agent is done
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
        logger.info("[TTS] Cleaning up TTS stream...")
        await tts.close()
        logger.info("[TTS] TTS stream cleanup complete")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, custom_prompt: str = None, language: str = "auto"):
    await websocket.accept()
    
    if custom_prompt:
        msg = f"Custom System Prompt Injected: {custom_prompt[:50]}..."
        bar = "*" * 50
        logger.info(f"\n{bar}\n{msg}\n{bar}")
        await websocket.send_json(event_to_dict(LogEvent.create(msg)))
    else:
        msg = f"Using Default System Prompt (language: {language})."
        logger.info(msg)
        await websocket.send_json(event_to_dict(LogEvent.create(msg)))

    # Only pass custom_prompt if it's actually provided
    # None means use default system_prompt with markdown formatting
    current_agent = get_agent(system_prompt_override=custom_prompt if custom_prompt else None, language=language)
    _agent_stream = make_agent_stream(current_agent)

    async def _stt_stream_local(
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[VoiceAgentEvent]:
        """
        Transform stream: Audio (Bytes) → Voice Events (VoiceAgentEvent)
        """
        # Determine language and model path for Whisper
        # "auto" -> None (Whisper detects)
        # "yue" (Cantonese) -> use custom model with auto-detect
        # "en", "zh", etc. -> passed directly
        
        whisper_lang = None if language == "auto" else language
        model_path = None 
        
        # Use custom Cantonese model if Cantonese is selected
        if language == "yue":
            model_path = "/home/robust/convert_faster_whisper/models/large-3-canto"
            whisper_lang = None  # Let the Cantonese model auto-detect (don't force "yue" code)
            logger.info(f"[System] Using custom Cantonese STT model: {model_path}")
        
        logger.info(f"[System] Initializing STT Model (Whisper) with language={whisper_lang}...")

        stt = LocalWhisperSTT(
            model_path=model_path,  # Use custom model if provided
            base_silence_threshold=300.0,  # Higher base threshold
            energy_window_size=5,
            speech_ratio_threshold=0.6,
            end_of_speech_silence=1.5,     # Wait 1.5s of silence before transcribing
            end_of_turn_silence=0.8,       # Extra 0.8s to confirm turn is over
            min_speech_duration=0.5,       # Lower min speech to catch short phrases
            max_buffer_duration=30.0,      # Allow up to 30s of speech
            use_noise_reduction=False,
            language=whisper_lang  
        )
        logger.info("[System] STT Model Ready.")

        async def send_audio():
            """
            Background task that pumps audio chunks to the STT model.
            """ 
            try: 
                # Stream each audio chunk to the STT as it arrives
                async for audio_chunk in audio_stream:
                    await stt.send_audio(audio_chunk)
            finally:
                # Signal that audio streaming is complete
                await stt.close()

        # Launch the audio sending task in the background
        send_task = asyncio.create_task(send_audio())

        try:
            # Consumer loop: receive and yield transcription events as they arrive
            async for event in stt.receive_events():
                yield event
        finally:
            # Cleanup: ensure proper shutdown
            logger.info("[STT] Cleaning up STT stream...")
            
            # First, close the STT to signal shutdown
            await stt.close()
            
            # Then cancel the send task if still running
            if not send_task.done():
                send_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await send_task
            
            logger.info("[STT] STT stream cleanup complete")

    pipeline = (
        RunnableGenerator(_stt_stream_local)  # Audio -> STT events
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
        # Clear Whisper model cache to reinitialize on next session
        clear_whisper_model_cache()
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
