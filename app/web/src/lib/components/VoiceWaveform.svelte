<script lang="ts">
  import { audioLevel } from "../stores/audioLevel";
  
  // Number of bars in the waveform visualization
  const BAR_COUNT = 24;
  
  // Keep track of level history for smooth animation
  let levelHistory: number[] = new Array(BAR_COUNT).fill(0);
  let animationFrame: number;
  
  // Subscribe to audio level and update history
  $: {
    const newLevel = $audioLevel;
    // Shift history left and add new level at the end
    levelHistory = [...levelHistory.slice(1), newLevel];
  }
  
  // Generate bar heights with some variation for visual interest
  function getBarHeights(): number[] {
    return levelHistory.map((level, i) => {
      // Add slight random variation for a more organic look
      const variation = 0.8 + Math.random() * 0.4;
      // Apply a wave pattern based on position
      const wave = Math.sin((i / BAR_COUNT) * Math.PI);
      return Math.max(0.1, level * variation * (0.5 + wave * 0.5));
    });
  }
  
  $: barHeights = getBarHeights();
</script>

<div class="voice-waveform">
  <div class="waveform-container">
    {#each barHeights as height, i}
      <div 
        class="bar"
        style="height: {Math.max(4, height * 48)}px; opacity: {0.4 + height * 0.6}"
      ></div>
    {/each}
  </div>
  <div class="level-indicator">
    <div class="level-fill" style="width: {$audioLevel * 100}%"></div>
  </div>
</div>

<style>
  .voice-waveform {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 12px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 12px;
    backdrop-filter: blur(4px);
  }
  
  .waveform-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 3px;
    height: 48px;
  }
  
  .bar {
    width: 4px;
    min-height: 4px;
    background: linear-gradient(to top, #22c55e, #4ade80);
    border-radius: 2px;
    transition: height 0.05s ease-out, opacity 0.1s ease-out;
  }
  
  .level-indicator {
    width: 100%;
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    overflow: hidden;
  }
  
  .level-fill {
    height: 100%;
    background: linear-gradient(to right, #22c55e, #4ade80);
    border-radius: 2px;
    transition: width 0.05s ease-out;
  }
</style>
