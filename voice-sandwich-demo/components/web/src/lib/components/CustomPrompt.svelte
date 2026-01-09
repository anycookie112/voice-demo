<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { logs } from "../stores";
  export let prompt = "";

  const dispatch = createEventDispatcher();

  let isConfirming = false;

  async function confirmPrompt() {
    if (!prompt.trim()) return;
    
    isConfirming = true;
    console.log("Custom prompt loaded:", prompt);
    logs.log(`Custom prompt loaded: ${prompt}`);
    dispatch("promptConfirmed", prompt);
    
    // Wait for animation, then clear
    await new Promise(resolve => setTimeout(resolve, 600));
    prompt = "";
    isConfirming = false;
  }
</script>

<div class="bg-white rounded-2xl p-6 mb-5 border border-gray-200">
  <label for="custom-prompt" class="block text-sm font-medium text-gray-500 mb-1">
    Custom System Prompt (Optional)
  </label>
  <div class="flex gap-2 flex-col">
    <textarea
      id="custom-prompt"
      bind:value={prompt}
      rows="5"
      class="w-full resize-none rounded bg-gray-50 border-gray-300 text-gray-900 text-sm focus:ring-indigo-500 focus:border-indigo-500 p-2 transition-all duration-300"
      class:opacity-50={isConfirming}
      placeholder={`Enter a custom system prompt to override the default behavior...

      Example:
      You are a Bread Seller in a Bakery called 7 Eleven.
      Users will ask you questions about bread and you can answer them accordingly.

      Bread A (Info)
      Bread B (Info)`}
    />
    <button
      on:click={confirmPrompt}
      disabled={isConfirming || !prompt.trim()}
      class="self-end px-4 py-2 bg-gray-900 hover:bg-gray-700 disabled:bg-gray-400 text-white text-sm font-medium rounded-md shadow-sm transition-all duration-200 flex items-center gap-2"
      class:scale-95={isConfirming}
    >
      {#if isConfirming}
        <svg class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        Confirming...
      {:else}
        Confirm Prompt
      {/if}
    </button>
  </div>
</div>


