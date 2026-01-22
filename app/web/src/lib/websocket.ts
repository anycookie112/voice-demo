import type { ServerEvent } from "./types";
import {
  session,
  currentTurn,
  latencyStats,
  waterfallData,
  activities,
  logs,
} from "./stores";
import { language } from "./stores/language";
import { createAudioCapture, createAudioPlayback } from "./audio";
import { get } from "svelte/store";

export interface VoiceSession {
  start: (customPrompt?: string) => Promise<void>;
  stop: () => void;
}

export function createVoiceSession(): VoiceSession {
  let ws: WebSocket | null = null;
  let ttsFinishTimeout: ReturnType<typeof setTimeout> | null = null;
  let activeMarkdownActivityId: string | null = null;
  let activeTtsActivityId: string | null = null;

  const audioCapture = createAudioCapture();
  const audioPlayback = createAudioPlayback();

  function handleEvent(event: ServerEvent) {
    const turn = get(currentTurn);

    switch (event.type) {
      case "stt_chunk":
        if (!turn.active) {
          // New turn - save previous waterfall data and reset
          const prevTurn = get(currentTurn);
          if (prevTurn.turnStartTs) {
            waterfallData.set({ ...prevTurn });
          }
          currentTurn.startTurn(event.ts);
          activeMarkdownActivityId = null; // Reset for new turn
          activeTtsActivityId = null;
        }
        currentTurn.sttStart(event.ts);
        currentTurn.sttChunk(event.transcript);
        break;

      case "stt_output":
        currentTurn.sttEnd(event.ts, event.transcript);
        activities.add("stt", "Transcription", event.transcript);
        break;

      case "agent_chunk": {
        currentTurn.agentChunk(event.ts, event.text);
        // Do not add to activities here; wait for agent_end
        break;
      }

      case "markdown_chunk": {
        // Display markdown content in activity feed
        if (activeMarkdownActivityId) {
          activities.appendToId(activeMarkdownActivityId, event.text);
        } else {
          activeMarkdownActivityId = activities.add("markdown", "Agent Response (Visual)", event.text);
        }
        break;
      }

      case "tts_text": {
        if (activeTtsActivityId) {
          activities.appendToId(activeTtsActivityId, event.text);
        } else {
          activeTtsActivityId = activities.add("tts", "Agent Response (Audio)", event.text);
        }
        break;
      }

      case "agent_end": {
        activeMarkdownActivityId = null;
        activeTtsActivityId = null;
        const currentTurnState = get(currentTurn);
        
        if (currentTurnState.response) {
          // STEP 1: Handle the removal logic carefully.
          // Only run removeLastOfType if you are SURE you added a 
          // temporary/loading bubble during the 'agent_chunk' or 'agent_start' phase.
          // If you strictly followed "Do not add to activities here", 
          // REMOVE the line below, or it will delete the Agent's PREVIOUS answer.
          if (activities.removeLastOfType) { 
             activities.removeLastOfType("agent"); 
          }

          // STEP 2: Commit the final message to the permanent history
          activities.add("agent", "Agent Response", currentTurnState.response);

          // STEP 3: THE FIX - Clear the streaming buffer!
          // You must reset currentTurn so the UI stops rendering the 
          // "streaming" version of the text alongside the "final" version.
          // (Use whatever method your store has to clear: .reset(), .clear(), or set to null)
          if (currentTurn.reset) {
            currentTurn.reset(); 
          } else {
            // Fallback if no reset method exists (adjust based on your state manager)
            // currentTurn.set({ response: "" }); 
          }
        }
        break;
      }
      

      case "tool_call":
        activities.add(
          "tool",
          `Tool: ${event.name}`,
          "Called with arguments:",
          event.args
        );
        logs.log(`Tool call: ${event.name}`);
        break;

      case "tool_result":
        activities.add("tool", `Tool Result: ${event.name}`, event.result);
        logs.log(`Tool result: ${event.result}`);
        break;


      case "tts_chunk": {
        const currentTurnState = get(currentTurn);

        console.log("Audio Chunk Received");
        console.log("Has TTS Started?", !!currentTurnState.ttsStartTs);
        console.log("Current Text Response:", currentTurnState.response);

        // Only update audio and state, do not add agent response here
        currentTurn.ttsChunk(event.ts);
        audioPlayback.push(event.audio);

        // Debounce: finish turn after TTS stops
        if (ttsFinishTimeout) clearTimeout(ttsFinishTimeout);
        ttsFinishTimeout = setTimeout(() => {
          const t = get(currentTurn);
          if (t.active && t.sttEndTs && t.ttsEndTs) {
            finishTurn();
          }
        }, 300);
        break;
      }
      
      case "log":
        logs.log(event.message);
        break;
      
      case "tts_end":
        console.log("TTS End Event Received");
        currentTurn.ttsEnd(event.ts);
        break;
      
      case "tts_stop":
        console.log("TTS Stop Event Received - Interrupting playback");
        audioPlayback.stop();
        break;
    //   case "tts_chunk": {
    //     // We don't need to add the activity here anymore because agent_chunk handles it.
    //     // However, if you want a fallback (in case audio arrives before text), use this:
    //     const currentTurnState = get(currentTurn);
        
    //     // Fallback: If we have text but no bubble yet, add it.
    //     // Note: Removed the '!ttsStartTs' check because it was causing the bug.
    //     if (currentTurnState.response && !hasBubbleBeenAdded(currentTurnState)) {
    //         activities.add("agent", "Agent Response", currentTurnState.response);
    //     }

    //     currentTurn.ttsChunk(event.ts);
    //     audioPlayback.push(event.audio);
    //     break;
    // }
    }
  }

  function finishTurn() {
    const turn = get(currentTurn);
    waterfallData.set({ ...turn });
    latencyStats.recordTurn(turn);
    currentTurn.finishTurn();
  }

  async function start(customPrompt?: string): Promise<void> {
    // Reset all state
    session.reset();
    currentTurn.reset();
    latencyStats.reset();
    waterfallData.set(null);
    activities.clear();
    activeMarkdownActivityId = null;
    activeTtsActivityId = null;
    logs.clear();
    audioPlayback.stop();

    session.setStatus("connecting");

    // Connect WebSocket
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const currentLanguage = get(language);
    const params = new URLSearchParams();
    if (customPrompt) {
      params.set("custom_prompt", customPrompt);
    }
    params.set("language", currentLanguage);
    const wsUrl = `${protocol}//${window.location.host}/ws?${params.toString()}`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = async () => {
      session.connect();
      logs.log("Session started");

      try {
        await audioCapture.start((chunk) => {
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(chunk);
          }
        });
        logs.log("Microphone access granted");
        logs.log("Streaming PCM audio (16kHz, 16-bit, mono)");
      } catch (err) {
        console.error(err);
        logs.log(
          `Error: ${err instanceof Error ? err.message : "Unknown error"}`
        );
        session.setStatus("error");
        stop();
      }
    };

    ws.onmessage = async (event) => {
      const eventData: ServerEvent = JSON.parse(event.data);
      handleEvent(eventData);
    };

    ws.onclose = () => {
      session.disconnect();
      logs.log("WebSocket disconnected");
    };

    ws.onerror = (e) => {
      console.error(e);
      logs.log("WebSocket error");
      session.setStatus("error");
    };
  }

  function stop(): void {
    logs.log("Session ended");

    if (ttsFinishTimeout) {
      clearTimeout(ttsFinishTimeout);
      ttsFinishTimeout = null;
    }

    audioPlayback.stop();
    audioCapture.stop();

    if (ws) {
      ws.close();
      ws = null;
    }

    session.reset();
  }

  return { start, stop };
}
