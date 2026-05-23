/**
 * Convert backend LLM messages (LiteLLM format) to Vercel AI SDK UIMessage format.
 */
import type { UIMessage } from 'ai';

interface LLMToolCall {
  id: string;
  function: { name: string; arguments: string };
}

interface LLMMessage {
  role: 'user' | 'assistant' | 'tool' | 'system';
  content: string | null;
  tool_calls?: LLMToolCall[] | null;
  tool_call_id?: string | null;
  name?: string | null;
}

let uiMessageCounter = 0;
function nextId(): string {
  return `msg-${++uiMessageCounter}`;
}

export function llmMessagesToUIMessages(
  messages: LLMMessage[],
  pendingApprovalIds?: Set<string>,
  existingUIMessages?: UIMessage[],
): UIMessage[] {
  const toolResults = new Map<string, { output: string; isError: boolean }>();
  for (const msg of messages) {
    if (msg.role === 'tool' && msg.tool_call_id) {
      toolResults.set(msg.tool_call_id, {
        output: msg.content || '',
        isError: false,
      });
    }
  }

  const uiMessages: UIMessage[] = [];
  const getExistingId = (index: number, role: 'user' | 'assistant'): string | null => {
    if (!existingUIMessages || index >= existingUIMessages.length) return null;
    const existing = existingUIMessages[index];
    return existing.role === role ? existing.id : null;
  };

  for (const msg of messages) {
    if (msg.role === 'system' || msg.role === 'tool') continue;

    if (msg.role === 'user') {
      if (typeof msg.content === 'string' && msg.content.trimStart().startsWith('[SYSTEM:')) {
        continue;
      }
      const existingId = getExistingId(uiMessages.length, 'user');
      uiMessages.push({
        id: existingId || nextId(),
        role: 'user',
        parts: [{ type: 'text', text: msg.content || '' }],
      });
      continue;
    }

    const parts: UIMessage['parts'] = [];
    if (msg.content) {
      parts.push({ type: 'text', text: msg.content });
    }

    if (msg.tool_calls) {
      for (const toolCall of msg.tool_calls) {
        let input: Record<string, unknown> = {};
        try {
          input = JSON.parse(toolCall.function.arguments);
        } catch {
          // Keep malformed tool-call arguments as an empty input object.
        }

        const result = toolResults.get(toolCall.id);
        if (result) {
          parts.push({
            type: 'dynamic-tool',
            toolCallId: toolCall.id,
            toolName: toolCall.function.name,
            state: 'output-available',
            input,
            output: result.output,
          });
        } else if (pendingApprovalIds?.has(toolCall.id)) {
          parts.push({
            type: 'dynamic-tool',
            toolCallId: toolCall.id,
            toolName: toolCall.function.name,
            state: 'approval-requested',
            input,
            approval: { id: `approval-${toolCall.id}` },
          });
        } else {
          parts.push({
            type: 'dynamic-tool',
            toolCallId: toolCall.id,
            toolName: toolCall.function.name,
            state: 'input-available',
            input,
          });
        }
      }
    }

    const previous = uiMessages[uiMessages.length - 1];
    if (previous && previous.role === 'assistant') {
      previous.parts.push(...parts);
    } else {
      const existingId = getExistingId(uiMessages.length, 'assistant');
      uiMessages.push({
        id: existingId || nextId(),
        role: 'assistant',
        parts,
      });
    }
  }

  return uiMessages;
}

interface ToolPart {
  type: string;
  toolCallId?: string;
  toolName?: string;
  state?: string;
  input?: unknown;
  output?: unknown;
  errorText?: string;
}

function joinText(parts: UIMessage['parts']): string {
  return parts
    .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
    .map((part) => part.text)
    .join('');
}

function stringifyOutput(output: unknown): string {
  if (output == null) return '';
  if (typeof output === 'string') return output;
  try {
    return JSON.stringify(output);
  } catch {
    return String(output);
  }
}

export function uiMessagesToLLMMessages(uiMessages: UIMessage[]): LLMMessage[] {
  const out: LLMMessage[] = [];
  for (const msg of uiMessages) {
    if (msg.role === 'user') {
      const text = joinText(msg.parts);
      if (text) out.push({ role: 'user', content: text });
      continue;
    }

    if (msg.role !== 'assistant') continue;

    const text = joinText(msg.parts);
    const toolCalls: LLMToolCall[] = [];
    const pairedResults: Array<{ id: string; content: string }> = [];
    for (const raw of msg.parts as ToolPart[]) {
      if (!raw.type) continue;
      const isTool = raw.type === 'dynamic-tool' || raw.type.startsWith('tool-');
      if (!isTool) continue;
      const toolCallId = raw.toolCallId;
      const toolName = raw.toolName ?? (raw.type.startsWith('tool-') ? raw.type.slice(5) : undefined);
      if (!toolCallId || !toolName) continue;

      toolCalls.push({
        id: toolCallId,
        function: {
          name: toolName,
          arguments: JSON.stringify(raw.input ?? {}),
        },
      });

      const result = raw.output != null
        ? stringifyOutput(raw.output)
        : typeof raw.errorText === 'string' && raw.errorText
          ? raw.errorText
          : null;
      if (result != null) {
        pairedResults.push({ id: toolCallId, content: result });
      }
    }

    if (text || toolCalls.length) {
      out.push({
        role: 'assistant',
        content: text || null,
        tool_calls: toolCalls.length ? toolCalls : null,
      });
    }
    for (const result of pairedResults) {
      out.push({ role: 'tool', content: result.content, tool_call_id: result.id });
    }
  }
  return out;
}
