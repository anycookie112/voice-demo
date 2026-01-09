<script lang="ts">
  import { Header, Controls, PipelineCard, ActivityFeed, Console, CustomPrompt } from './lib/components';
  import { createVoiceSession } from './lib/websocket';

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
  <PipelineCard />
  <ActivityFeed />
  <Console />
</div>

