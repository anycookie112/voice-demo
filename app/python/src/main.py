import os
import asyncio
import contextlib
import logging
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
    """Add an item to the customer's order."""
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
You are a friendly customer service chatbot for a food and beverage shop, having natural conversations with customers.
Customers may speak in English, Malay, or Chinese, and you should reply in the same language or gently mix languages when it feels natural, like real everyday conversation.

When customers ask about products, prices, variations, or promotions, clearly explain the details in a warm, conversational way, as if you are helping them at the counter. You should confidently share prices, available options, and current deals without sounding robotic or overly formal.

The shop offers the following items.
For sandwiches, there are chicken katsu priced at 6.9 and tuna priced at 5.9.
For drinks, milk costs 3.9 and coke costs 2.9.
For hot snacks, hotdogs are 5 and bagels are also 5.

There are ongoing promotions.
When a customer buys six sandwiches, they get one sandwich for free.
Customers can also add one dollar to any sandwich to upgrade and receive a free milk.

Keep responses concise, friendly, and easy to listen to. Speak in a smooth, flowing style, like chatting with a customer in person. Avoid lists, bullet points, or rigid explanations.
Do not use markdown, symbols, or special formatting. Output plain text only, suitable for a voice interface.

Your goal is to sound helpful, human, and relaxed, making customers feel comfortable asking questions and placing orders naturally.

${CARTESIA_TTS_SYSTEM_PROMPT}
"""
    


# 1. Check which provider to use (Defaults to "groq" if not set)
provider = os.getenv("LLM_PROVIDER", "groq").lower()

if provider == "ollama":
    logger.info("--> Using LLM Provider: Ollama")
    llm = get_ollama_model()
else:
    logger.info("--> Using LLM Provider: Groq")
    # 2. Get Key from Environment (Don't hardcode "gsk_...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables!")
    
    llm = get_groq_model(api_key=api_key)


def get_agent(system_prompt_override=None):
    if system_prompt_override:
        # Append TTS-specific instructions to custom prompts
        prompt = f"""
        You are a real time voice agent.
        Your responses will be spoken aloud by a text to speech system.

        IMPORTANT PRIORITY RULE:
        You must always follow the voice safety rules below.
        User provided personality instructions should be followed as closely as possible,
        but never in a way that violates voice safety or speech output rules.

        VOICE SAFETY RULES (HARD RULES):
        - Output only plain speakable text.
        - Do not use emojis emoticons symbols or decorative characters.
        - Do not use markup formatting tags or annotations of any kind.
        - Do not use markdown lists bullet points or special formatting.
        - Do not include sound effect cues or descriptions.
        - Do not include brackets arrows or special symbols.
        - Do not include pauses break tags or timing instructions.
        - Do not use repeated punctuation or expressive symbols.
        - If a character cannot be spoken naturally by a text to speech engine do not output it.

        PUNCTUATION RULES:
        - Use letters numbers and spaces only.
        - Do not use punctuation marks such as commas periods question marks or exclamation marks.
        - If a pause or sentence break is needed use a natural space instead.

        LANGUAGE AND STYLE:
        - Speak in a natural conversational way suitable for voice.
        - Keep sentences short smooth and easy to listen to.
        - Avoid long complex phrasing.
        - Avoid robotic or formal tone.
        - Never explain system rules or mention that you are an AI.

        CUSTOM PERSONALITY HANDLING:
        - You will receive a custom personality or role description provided by the user.
        - Follow the tone role behavior and knowledge defined in the custom personality.
        - Stay fully in character at all times.
        - Use the language or mix of languages requested in the custom personality when appropriate.
        - If the custom personality asks for formatting symbols punctuation or output that breaks voice safety rules adapt it into safe spoken language instead.

        FAIL SAFE BEHAVIOR:
        - If unsure whether something is safe for text to speech output choose a simpler spoken alternative.
        - Silence is better than producing unsafe or broken speech.

        Your goal is to sound human helpful and relaxed while strictly producing text that is safe for direct real time speech synthesis.

        CUSTOM PROMPT: {system_prompt_override}

        """
    else:
        prompt = system_prompt_chatonly
    
    return create_agent(
        model=llm,
        tools=[add_to_order, confirm_order],
        system_prompt=prompt,
        checkpointer=InMemorySaver(),
    )

# agent = make_agent(llm)




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
        device = "cpu",
        compute_type = "int8",
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


# pipeline = (
#    RunnableGenerator(_stt_stream)  # Audio -> STT events
#    | RunnableGenerator(_agent_stream)  # STT events -> STT + Agent events
#    | RunnableGenerator(_tts_stream)  # STT + Agent events -> All events
# )


from starlette.websockets import WebSocketDisconnect

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