/**
 * Event types from the agent backend
 */

export type EventType =
  | 'ready'
  | 'processing'
  | 'assistant_message'
  | 'assistant_chunk'
  | 'assistant_stream_end'
  | 'tool_call'
  | 'tool_output'
  | 'tool_log'
  | 'approval_required'
  | 'auto_finetune_progress'
  | 'tool_state_change'
  | 'turn_complete'
  | 'compacted'
  | 'error'
  | 'shutdown'
  | 'interrupted'
  | 'undo_complete'
  | 'plan_update';

export interface AgentEvent {
  event_type: EventType;
  seq?: number;
  data?: Record<string, unknown>;
}

export interface ReadyEventData {
  message: string;
}

export interface ProcessingEventData {
  message: string;
}

export interface AssistantMessageEventData {
  content: string;
}

export interface ToolCallEventData {
  tool: string;
  arguments: Record<string, unknown>;
}

export interface ToolOutputEventData {
  tool: string;
  output: string;
  success: boolean;
}

export interface ToolLogEventData {
  tool: string;
  log: string;
}

export interface PlanUpdateEventData {
  plan: Array<{ id: string; content: string; status: 'pending' | 'in_progress' | 'completed' }>;
}

export interface ApprovalRequiredEventData {
  tools: ApprovalToolItem[];
  count: number;
}

export interface AutoFineTuneProgressEventData {
  state:
    | 'resolving_dataset'
    | 'preflight_passed'
    | 'submitting_job'
    | 'retrying_job'
    | 'completed'
    | 'blocked'
    | 'failed'
    | string;
  message: string;
  provider_id: 'hf-jobs';
  approval_required: false;
  credential_readiness?: Record<string, boolean>;
  model_repo_url?: string;
  job_url?: string;
  eval_result?: string;
  estimated_total_cost_usd?: number;
  cost_cap_usd?: number;
  attempt?: number;
  max_attempts?: number;
  error_code?: string;
}

export interface ApprovalToolItem {
  tool: string;
  arguments: Record<string, unknown>;
  tool_call_id: string;
}

export interface TurnCompleteEventData {
  history_size: number;
}

export interface CompactedEventData {
  old_tokens: number;
  new_tokens: number;
}

export interface ErrorEventData {
  error: string;
}
