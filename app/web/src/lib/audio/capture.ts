// AudioWorklet code for PCM capture
const workletCode = `
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.targetSampleRate = 16000;
    this.resampleRatio = sampleRate / this.targetSampleRate;
    this.resampleIndex = 0;
    this.levelAccumulator = 0;
    this.levelSampleCount = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];

    // Calculate RMS level for visualization
    let sumSquares = 0;
    for (let i = 0; i < channelData.length; i++) {
      sumSquares += channelData[i] * channelData[i];
    }
    const rms = Math.sqrt(sumSquares / channelData.length);
    this.levelAccumulator += rms;
    this.levelSampleCount++;

    // Send level update every ~50ms (about 4 process calls at typical buffer sizes)
    if (this.levelSampleCount >= 4) {
      const avgLevel = this.levelAccumulator / this.levelSampleCount;
      this.port.postMessage({ type: 'level', level: avgLevel });
      this.levelAccumulator = 0;
      this.levelSampleCount = 0;
    }

    for (let i = 0; i < channelData.length; i++) {
      this.resampleIndex += 1;
      if (this.resampleIndex >= this.resampleRatio) {
        this.resampleIndex -= this.resampleRatio;
        let sample = channelData[i];
        sample = Math.max(-1, Math.min(1, sample));
        const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        this.buffer.push(int16);
      }
    }

    const CHUNK_SIZE = 1600;
    while (this.buffer.length >= CHUNK_SIZE) {
      const chunk = this.buffer.splice(0, CHUNK_SIZE);
      const int16Array = new Int16Array(chunk);
      this.port.postMessage({ type: 'audio', data: int16Array.buffer }, [int16Array.buffer]);
    }

    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
`;

import { audioLevel } from "../stores/audioLevel";

export interface AudioCapture {
  start: (onChunk: (chunk: ArrayBuffer) => void) => Promise<void>;
  stop: () => void;
}

export function createAudioCapture(): AudioCapture {
  let audioContext: AudioContext | null = null;
  let workletNode: AudioWorkletNode | null = null;
  let mediaStream: MediaStream | null = null;
  let sourceNode: MediaStreamAudioSourceNode | null = null;

  async function start(onChunk: (chunk: ArrayBuffer) => void): Promise<void> {
    // Always create a fresh AudioContext to ensure clean state
    // Close previous one if it exists
    if (audioContext) {
      try {
        await audioContext.close();
      } catch (e) {
        console.warn("Error closing previous AudioContext:", e);
      }
      audioContext = null;
    }

    audioContext = new AudioContext();
    const blob = new Blob([workletCode], { type: "application/javascript" });
    const workletUrl = URL.createObjectURL(blob);
    await audioContext.audioWorklet.addModule(workletUrl);
    URL.revokeObjectURL(workletUrl);

    // Resume if suspended
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    // Get microphone access
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    // Create worklet node and connect
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioContext, "pcm-processor");


    workletNode.port.onmessage = (event) => {
      const data = event.data;
      if (data.type === 'level') {
        // Update the audio level store for visualization
        audioLevel.set(Math.min(1, data.level * 3)); // Amplify for visibility
      } else if (data.type === 'audio') {
        onChunk(data.data);
      }
    };

    sourceNode.connect(workletNode);
  }

  function stop(): void {
    // Reset audio level to zero immediately
    audioLevel.set(0);

    if (workletNode) {
      workletNode.disconnect();
      workletNode.port.onmessage = null;
      workletNode = null;
    }

    if (sourceNode) {
      sourceNode.disconnect();
      sourceNode = null;
    }

    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }

    // Close the AudioContext to fully reset state
    if (audioContext) {
      audioContext.close().catch(console.warn);
      audioContext = null;
    }
  }

  return { start, stop };
}
