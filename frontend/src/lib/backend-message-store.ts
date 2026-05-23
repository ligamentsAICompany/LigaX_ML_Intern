/**
 * localStorage cache of raw backend (litellm Message) dicts keyed by
 * session ID. Used to restore a session into a fresh backend after restart.
 */
import { logger } from '@/utils/logger';

const STORAGE_KEY = 'hf-agent-backend-messages';
const MAX_SESSIONS = 50;

type MessagesMap = Record<string, unknown[]>;

function readAll(): MessagesMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      return parsed as MessagesMap;
    }
    return {};
  } catch {
    return {};
  }
}

function writeAll(map: MessagesMap): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch (error) {
    logger.warn('Failed to persist backend messages:', error);
  }
}

export function loadBackendMessages(sessionId: string): unknown[] {
  return readAll()[sessionId] ?? [];
}

export function saveBackendMessages(sessionId: string, messages: unknown[]): void {
  const map = readAll();
  map[sessionId] = messages;
  const keys = Object.keys(map);
  if (keys.length > MAX_SESSIONS) {
    for (const key of keys.slice(0, keys.length - MAX_SESSIONS)) delete map[key];
  }
  writeAll(map);
}

export function moveBackendMessages(fromId: string, toId: string): void {
  const map = readAll();
  if (!map[fromId]) return;
  map[toId] = map[fromId];
  delete map[fromId];
  writeAll(map);
}

export function deleteBackendMessages(sessionId: string): void {
  const map = readAll();
  delete map[sessionId];
  writeAll(map);
}
