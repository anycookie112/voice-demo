import { writable } from "svelte/store";
import type { ActivityItem, LogEntry } from "../types";

function createActivityStore() {
  const { subscribe, update, set } = writable<ActivityItem[]>([]);

  let idCounter = 0;

  return {
    subscribe,

    add(
      type: ActivityItem["type"],
      label: string,
      text: string,
      args?: Record<string, unknown>
    ) {
      const id = `activity-${++idCounter}`;
      const item: ActivityItem = {
        id,
        type,
        label,
        text,
        args,
        timestamp: new Date(),
      };
      update((items) => [item, ...items].slice(0, 50)); // Keep max 50 items
      return id;
    },

    removeLastOfType(type: ActivityItem["type"]) {
      update((items) => {
        const idx = items.findIndex((item) => item.type === type);
        if (idx !== -1) {
          return [...items.slice(0, idx), ...items.slice(idx + 1)];
        }
        return items;
      });
    },

    appendToId(id: string, text: string) {
      update((items) => 
        items.map((item) => 
          item.id === id ? { ...item, text: item.text + text } : item
        )
      );
    },

    clear() {
      set([]);
    },
  };
}

function createLogStore() {
  const { subscribe, update, set } = writable<LogEntry[]>([]);

  let idCounter = 0;

  return {
    subscribe,

    log(message: string) {
      const entry: LogEntry = {
        id: `log-${++idCounter}`,
        message,
        timestamp: new Date(),
      };
      update((entries) => [...entries, entry].slice(-100)); // Keep max 100 entries
    },

    clear() {
      set([]);
    },
  };
}

export const activities = createActivityStore();
export const logs = createLogStore();
