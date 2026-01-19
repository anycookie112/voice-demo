import { writable } from "svelte/store";

// Store for real-time audio level (0-1)
export const audioLevel = writable(0);

// Store for audio level history (for waveform visualization)
export const audioLevelHistory = writable<number[]>(new Array(50).fill(0));

let historyUpdateInterval: ReturnType<typeof setInterval> | null = null;

export function startAudioLevelTracking() {
  // Update history every 50ms for smooth waveform
  historyUpdateInterval = setInterval(() => {
    audioLevelHistory.update((history) => {
      let currentLevel = 0;
      audioLevel.subscribe((v) => (currentLevel = v))();
      return [...history.slice(1), currentLevel];
    });
  }, 50);
}

export function stopAudioLevelTracking() {
  if (historyUpdateInterval) {
    clearInterval(historyUpdateInterval);
    historyUpdateInterval = null;
  }
  audioLevel.set(0);
  audioLevelHistory.set(new Array(50).fill(0));
}
