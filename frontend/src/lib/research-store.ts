/**
 * Persist research sub-agent state per session so refreshes keep recent steps.
 */
import type { PerSessionState } from '@/store/agentStore';

export const RESEARCH_MAX_STEPS = 4;

const STORAGE_KEY = 'hf-agent-research';

type ResearchState = {
  steps: string[];
  stats: PerSessionState['researchStats'];
};

type ResearchMap = Record<string, ResearchState>;

function readAll(): ResearchMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function writeAll(map: ResearchMap): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch {
    // Best-effort UI cache.
  }
}

export function saveResearch(
  sessionId: string,
  steps: string[],
  stats: PerSessionState['researchStats'],
): void {
  const map = readAll();
  map[sessionId] = {
    steps: steps.slice(-RESEARCH_MAX_STEPS),
    stats,
  };
  writeAll(map);
}

export function loadResearch(sessionId: string): ResearchState | null {
  return readAll()[sessionId] ?? null;
}

export function clearResearch(sessionId: string): void {
  const map = readAll();
  delete map[sessionId];
  writeAll(map);
}
