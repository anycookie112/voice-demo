<script lang="ts">
  import { Header, Controls, ActivityFeed, Console, CustomPrompt, VoiceWaveform } from './lib/components';
  import { createVoiceSession } from './lib/websocket';
  import { session } from './lib/stores';

  const voiceSession = createVoiceSession();
  let customPromptInput = "";
  let confirmedPrompt = "";

  function handlePromptConfirmed(event: CustomEvent<string>) {
    confirmedPrompt = event.detail;
    console.log("Prompt confirmed and stored:", confirmedPrompt);
  }
</script>

<div class="max-w-3xl mx-auto">
  <Header />
  <CustomPrompt bind:prompt={customPromptInput} on:promptConfirmed={handlePromptConfirmed} />
  <Controls onStart={() => voiceSession.start(confirmedPrompt)} onStop={() => voiceSession.stop()} />
  
  <div class="mb-4 transition-opacity duration-300" class:opacity-40={!$session.connected} class:grayscale={!$session.connected}>
    <VoiceWaveform />
  </div>
  
  <!-- <PipelineCard /> -->
  <ActivityFeed />
  <Console />
</div>

