import { writable } from "svelte/store";

export type LanguageOption = "auto" | "en" | "zh" | "yue" | "ms";

export const languageLabels: Record<LanguageOption, string> = {
  auto: "AUTO",
  en: "EN",
  zh: "CN",
  yue: "粵",
  ms: "MY",
};

function createLanguageStore() {
  const { subscribe, set, update } = writable<LanguageOption>("auto");

  return {
    subscribe,
    set,
    setLanguage(lang: LanguageOption) {
      set(lang);
    },
  };
}

export const language = createLanguageStore();
