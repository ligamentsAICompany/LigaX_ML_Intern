/**
 * SSE-based ChatTransport that bridges backend events to Vercel AI SDK chunks.
 */
import type { ChatRequestOptions, ChatTransport, UIMessage, UIMessageChunk } from 'ai';
import { apiFetch } from '@/utils/api';
import { logger } from '@/utils/logger';
import type { AgentEvent } from '@/types/events';
import { useAgentStore } from '@/store/agentStore';

export interface SideChannelCallbacks {
  onReady: () => void;
  onShutdown: () => void;
  onError: (error: string) => void;
  onProcessing: () => void;
  onProcessingDone: () => void;
  onUndoComplete: () => void;
  onCompacted: (oldTokens: number, newTokens: number) => void;
  onPlanUpdate: (plan: Array<{ id: string; content: string; status: string }>) => void;
  onToolLog: (tool: string, log: string, agentId?: string, label?: string) => void;
  onConnectionChange: (connected: boolean) => void;
  onSessionDead: (sessionId: string) => void;
  onApprovalRequired: (tools: Array<{
    tool: string;
    arguments: Record<string, unknown>;
    tool_call_id: string;
    auto_approval_blocked?: boolean;
    block_reason?: string | null;
    estimated_cost_usd?: number | null;
    remaining_cap_usd?: number | null;
  }>) => void;
  onToolCallPanel: (tool: string, args: Record<string, unknown>) => void;
  onToolOutputPanel: (tool: string, toolCallId: string, output: string, success: boolean) => void;
  onStreaming: () => void;
  onToolRunning: (toolName: string, description?: string) => void;
  onInterrupted: () => void;
}

let partIdCounter = 0;
function nextPartId(prefix: string): string {
  return `${prefix}-${Date.now()}-${++partIdCounter}`;
}

function lastEventKey(sessionId: string): string {
  return `hf-agent-last-event:${sessionId}`;
}

function createSSEParserStream(sessionId: string): TransformStream<string, AgentEvent> {
  let buffer = '';
  let eventId: string | null = null;
  let data = '';

  const dispatch = (controller: TransformStreamDefaultController<AgentEvent>) => {
    if (!data.trim()) {
      eventId = null;
      data = '';
      return;
    }
    try {
      const json = JSON.parse(data.trim()) as AgentEvent;
      const seq = json.seq ?? (eventId ? Number(eventId) : undefined);
      if (Number.isFinite(seq)) {
        json.seq = seq;
        localStorage.setItem(lastEventKey(sessionId), String(seq));
      }
      controller.enqueue(json);
    } catch {
      logger.warn('SSE parse error:', data.trim());
    } finally {
      eventId = null;
      data = '';
    }
  };

  return new TransformStream<string, AgentEvent>({
    transform(chunk, controller) {
      buffer += chunk;
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const rawLine of lines) {
        const line = rawLine.replace(/\r$/, '');
        if (line === '') {
          dispatch(controller);
        } else if (line.startsWith('id:')) {
          eventId = line.slice(3).trim();
        } else if (line.startsWith('data:')) {
          data += `${line.slice(5).trimStart()}\n`;
        }
      }
    },
    flush(controller) {
      const line = buffer.replace(/\r$/, '');
      if (line.startsWith('id:')) eventId = line.slice(3).trim();
      if (line.startsWith('data:')) data += `${line.slice(5).trimStart()}\n`;
      dispatch(controller);
    },
  });
}

function createEventToChunkStream(sideChannel: SideChannelCallbacks): TransformStream<AgentEvent, UIMessageChunk> {
  let textPartId: string | null = null;

  function endTextPart(controller: TransformStreamDefaultController<UIMessageChunk>) {
    if (!textPartId) return;
    controller.enqueue({ type: 'text-end', id: textPartId });
    textPartId = null;
  }

  return new TransformStream<AgentEvent, UIMessageChunk>({
    transform(event, controller) {
      switch (event.event_type) {
        case 'ready':
          sideChannel.onReady();
          break;
        case 'shutdown':
          endTextPart(controller);
          controller.enqueue({ type: 'finish-step' });
          controller.enqueue({ type: 'finish', finishReason: 'stop' });
          sideChannel.onShutdown();
          break;
        case 'interrupted':
          endTextPart(controller);
          controller.enqueue({ type: 'finish-step' });
          controller.enqueue({ type: 'finish', finishReason: 'stop' });
          sideChannel.onInterrupted();
          sideChannel.onProcessingDone();
          break;
        case 'undo_complete':
          endTextPart(controller);
          sideChannel.onUndoComplete();
          break;
        case 'compacted':
          sideChannel.onCompacted(
            (event.data?.old_tokens as number) || 0,
            (event.data?.new_tokens as number) || 0,
          );
          break;
        case 'plan_update':
          sideChannel.onPlanUpdate((event.data?.plan as Array<{ id: string; content: string; status: string }>) || []);
          break;
        case 'tool_log':
          sideChannel.onToolLog(
            (event.data?.tool as string) || '',
            (event.data?.log as string) || '',
            (event.data?.agent_id as string) || '',
            (event.data?.label as string) || '',
          );
          break;
        case 'processing':
          sideChannel.onProcessing();
          controller.enqueue({ type: 'start', messageMetadata: { createdAt: new Date().toISOString() } });
          controller.enqueue({ type: 'start-step' });
          break;
        case 'assistant_chunk': {
          const delta = (event.data?.content as string) || '';
          if (!delta) break;
          if (!textPartId) {
            textPartId = nextPartId('text');
            controller.enqueue({ type: 'text-start', id: textPartId });
            sideChannel.onStreaming();
          }
          controller.enqueue({ type: 'text-delta', id: textPartId, delta });
          break;
        }
        case 'assistant_stream_end':
          endTextPart(controller);
          break;
        case 'assistant_message': {
          const content = (event.data?.content as string) || '';
          if (!content) break;
          const id = nextPartId('text');
          controller.enqueue({ type: 'text-start', id });
          controller.enqueue({ type: 'text-delta', id, delta: content });
          controller.enqueue({ type: 'text-end', id });
          break;
        }
        case 'tool_call': {
          const toolName = (event.data?.tool as string) || 'unknown';
          const toolCallId = (event.data?.tool_call_id as string) || '';
          const args = (event.data?.arguments as Record<string, unknown>) || {};
          if (toolName === 'plan_tool') break;
          endTextPart(controller);
          controller.enqueue({ type: 'tool-input-start', toolCallId, toolName, dynamic: true });
          controller.enqueue({ type: 'tool-input-available', toolCallId, toolName, input: args, dynamic: true });
          sideChannel.onToolRunning(toolName, args.description as string | undefined);
          sideChannel.onToolCallPanel(toolName, args);
          break;
        }
        case 'tool_output': {
          const toolCallId = (event.data?.tool_call_id as string) || '';
          const output = (event.data?.output as string) || '';
          const success = event.data?.success as boolean;
          const toolName = (event.data?.tool as string) || '';
          if (toolName === 'plan_tool' || toolCallId.startsWith('plan_tool')) break;
          controller.enqueue(success
            ? { type: 'tool-output-available', toolCallId, output, dynamic: true }
            : { type: 'tool-output-error', toolCallId, errorText: output, dynamic: true });
          sideChannel.onToolOutputPanel(toolName, toolCallId, output, success);
          break;
        }
        case 'approval_required': {
          const tools = event.data?.tools as Array<{
            tool: string;
            arguments: Record<string, unknown>;
            tool_call_id: string;
            auto_approval_blocked?: boolean;
            block_reason?: string | null;
            estimated_cost_usd?: number | null;
            remaining_cap_usd?: number | null;
          }> | undefined;
          if (!tools) break;
          endTextPart(controller);
          for (const tool of tools) {
            controller.enqueue({ type: 'tool-input-start', toolCallId: tool.tool_call_id, toolName: tool.tool, dynamic: true });
            controller.enqueue({ type: 'tool-input-available', toolCallId: tool.tool_call_id, toolName: tool.tool, input: tool.arguments, dynamic: true });
            controller.enqueue({ type: 'tool-approval-request', approvalId: `approval-${tool.tool_call_id}`, toolCallId: tool.tool_call_id });
          }
          sideChannel.onApprovalRequired(tools);
          break;
        }
        case 'tool_state_change': {
          const toolCallId = (event.data?.tool_call_id as string) || '';
          const state = (event.data?.state as string) || '';
          const toolName = (event.data?.tool as string) || '';
          const jobUrl = (event.data?.jobUrl as string) || undefined;
          const trackioSpaceId = (event.data?.trackioSpaceId as string) || undefined;
          const trackioProject = (event.data?.trackioProject as string) || undefined;
          if (toolCallId.startsWith('plan_tool')) break;
          if (jobUrl && toolCallId) useAgentStore.getState().setJobUrl(toolCallId, jobUrl);
          if (trackioSpaceId && toolCallId) useAgentStore.getState().setTrackioDashboard(toolCallId, trackioSpaceId, trackioProject);
          if (state === 'running' && toolName) sideChannel.onToolRunning(toolName);
          if (state === 'rejected' || state === 'abandoned') controller.enqueue({ type: 'tool-output-denied', toolCallId });
          if (state === 'cancelled') controller.enqueue({ type: 'tool-output-error', toolCallId, errorText: 'Cancelled by user', dynamic: true });
          if (state === 'billing_required') {
            const namespace = (event.data?.namespace as string) || '';
            useAgentStore.getState().setJobsUpgradeRequired({
              namespace: namespace || null,
              message: namespace
                ? `Hugging Face Jobs need credits on the "${namespace}" namespace. Job credits are separate from HF Pro membership; add credits, then re-run the same job.`
                : 'Hugging Face Jobs need namespace credits, which are separate from HF Pro membership. Add credits, then re-run the same job.',
            });
          }
          break;
        }
        case 'turn_complete':
          endTextPart(controller);
          controller.enqueue({ type: 'finish-step' });
          controller.enqueue({ type: 'finish', finishReason: 'stop' });
          sideChannel.onProcessingDone();
          break;
        case 'error': {
          const errorMsg = (event.data?.error as string) || 'Unknown error';
          endTextPart(controller);
          controller.enqueue({ type: 'finish-step' });
          controller.enqueue({ type: 'finish', finishReason: 'error' });
          sideChannel.onError(errorMsg);
          sideChannel.onProcessingDone();
          break;
        }
        default:
          logger.log('SSE transport: unknown event', event);
      }
    },
  });
}

export class SSEChatTransport implements ChatTransport<UIMessage> {
  private sessionId: string;
  private sideChannel: SideChannelCallbacks;

  constructor(sessionId: string, sideChannel: SideChannelCallbacks) {
    this.sessionId = sessionId;
    this.sideChannel = sideChannel;
    queueMicrotask(() => sideChannel.onConnectionChange(true));
  }

  updateSideChannel(sideChannel: SideChannelCallbacks): void {
    this.sideChannel = sideChannel;
  }

  destroy(): void {
    // No persistent connection to close.
  }

  async sendMessages(
    options: {
      trigger: 'submit-message' | 'regenerate-message';
      chatId: string;
      messageId: string | undefined;
      messages: UIMessage[];
      abortSignal: AbortSignal | undefined;
    } & ChatRequestOptions,
  ): Promise<ReadableStream<UIMessageChunk>> {
    const lastAssistant = [...options.messages].reverse().find((message) => message.role === 'assistant');
    const approvedParts = lastAssistant?.parts.filter(
      (part) => part.type === 'dynamic-tool' && part.state === 'approval-responded',
    ) || [];

    let body: Record<string, unknown>;
    if (approvedParts.length > 0) {
      const approvals = approvedParts.map((part) => {
        if (part.type !== 'dynamic-tool') return null;
        const approved = part.approval?.approved ?? true;
        const editedScript = useAgentStore.getState().getEditedScript(part.toolCallId);
        return {
          tool_call_id: part.toolCallId,
          approved,
          feedback: approved ? null : (part.approval?.reason || 'Rejected by user'),
          edited_script: editedScript ?? null,
          namespace: null,
        };
      }).filter(Boolean);
      body = { approvals };
    } else {
      const lastUserMsg = [...options.messages].reverse().find((message) => message.role === 'user');
      const text = lastUserMsg
        ? lastUserMsg.parts
          .filter((part): part is Extract<typeof part, { type: 'text' }> => part.type === 'text')
          .map((part) => part.text)
          .join('')
        : '';
      body = { text };
    }

    const response = await apiFetch(`/api/chat/${this.sessionId}`, {
      method: 'POST',
      body: JSON.stringify(body),
      signal: options.abortSignal,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
    });

    if (response.status === 404) this.sideChannel.onSessionDead(this.sessionId);
    if (response.status === 429) throw new Error('CLAUDE_QUOTA_EXHAUSTED');
    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Request failed');
      throw new Error(`Chat request failed: ${response.status} ${errorText}`);
    }
    if (!response.body) throw new Error('No response body');

    return response.body
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(createSSEParserStream(this.sessionId))
      .pipeThrough(createEventToChunkStream(this.sideChannel));
  }

  async reconnectToStream(): Promise<ReadableStream<UIMessageChunk> | null> {
    try {
      const infoRes = await apiFetch(`/api/session/${this.sessionId}`);
      if (!infoRes.ok) return null;
      const info = await infoRes.json();
      if (!info.is_processing) return null;

      const lastSeq = localStorage.getItem(lastEventKey(this.sessionId));
      const qs = lastSeq ? `?after=${encodeURIComponent(lastSeq)}` : '';
      const response = await apiFetch(`/api/events/${this.sessionId}${qs}`, {
        headers: { Accept: 'text/event-stream' },
      });
      if (!response.ok || !response.body) return null;

      this.sideChannel.onProcessing();
      return response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(createSSEParserStream(this.sessionId))
        .pipeThrough(createEventToChunkStream(this.sideChannel));
    } catch {
      return null;
    }
  }
}
