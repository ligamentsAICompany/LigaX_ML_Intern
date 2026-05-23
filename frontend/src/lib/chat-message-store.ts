/**
 * Lightweight localStorage persistence for UIMessage arrays keyed by session ID.
 */
import type { UIMessage } from 'ai';
import { logger } from '@/utils/logger';

const STORAGE_KEY = 'hf-agent-messages';
const MAX_SESSIONS = 50;

type MessagesMap = Record<string, UIMessage[]>;

function readAll(): MessagesMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (parsed.messagesBySession) return parsed.messagesBySession;
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
    logger.warn('Failed to persist messages:', error);
  }
}

export function loadMessages(sessionId: string): UIMessage[] {
  return readAll()[sessionId] ?? [];
}

export function saveMessages(sessionId: string, messages: UIMessage[]): void {
  const map = readAll();
  map[sessionId] = messages;
  const keys = Object.keys(map);
  if (keys.length > MAX_SESSIONS) {
    for (const key of keys.slice(0, keys.length - MAX_SESSIONS)) delete map[key];
  }
  writeAll(map);
}

export function deleteMessages(sessionId: string): void {
  const map = readAll();
  delete map[sessionId];
  writeAll(map);
}

export function moveMessages(fromId: string, toId: string): void {
  const map = readAll();
  if (!map[fromId]) return;
  map[toId] = map[fromId];
  delete map[fromId];
  writeAll(map);
}
