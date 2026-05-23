/**
 * Agent-related types.
 *
 * Message and tool-call types are now provided by the Vercel AI SDK
 * (UIMessage, UIMessagePart, etc.). Only non-SDK types remain here.
 */

/** Custom metadata attached to every UIMessage via the `metadata` field. */
export interface MessageMeta {
  createdAt?: string;
}

export type DomainId = 'generic' | 'itr' | 'gst' | 'fieldops' | 'call-center';

export type CloudProviderId = 'hf-jobs' | 'aws' | 'azure' | 'gcp';

export type DatasetUploadStatus = 'idle' | 'uploading' | 'ready' | 'error';

export type MLQualityRiskLevel = 'low' | 'medium' | 'high';

export type MLStrategyRecommendation = 'fine_tune' | 'rag' | 'hybrid' | 'data_needed';

export interface TrainabilitySummary {
  score?: number;
  recommendation?: MLStrategyRecommendation;
  risk_level?: MLQualityRiskLevel;
  reasons?: string[];
}

export interface StrategySummary {
  strategy?: MLStrategyRecommendation;
  confidence?: number;
  risk_level?: MLQualityRiskLevel;
  reasons?: string[];
  required_next_actions?: string[];
  can_train_without_override?: boolean;
  requires_user_override_for_training?: boolean;
  method_hint?: string | null;
  override_message?: string;
}

export interface GoldenEvalCase {
  id?: string;
  question?: string;
  expected_answer?: string;
  task_type?: string;
  source_metadata?: Record<string, unknown>;
}

export interface GoldenEvalSummary {
  case_count?: number;
  quality_constraints?: string[];
  cases?: GoldenEvalCase[];
}

export interface ReferenceLookupSummary {
  ready?: boolean;
  row_count?: number;
  key_columns?: string[];
  answer_columns?: string[];
  status?: string;
}

export interface PostTrainingEvalSummary {
  status?: string;
  valid?: boolean;
  case_count?: number;
  passed?: number;
  failed?: number;
  needs_rag?: number;
  needs_more_data?: number;
  top_reasons?: string[];
  summary?: Record<string, number>;
}

export type AutoFineTuneProgressState =
  | 'resolving_dataset'
  | 'preflight_passed'
  | 'submitting_job'
  | 'retrying_job'
  | 'completed'
  | 'blocked'
  | 'failed'
  | string;

export interface AutoFineTuneProgress {
  state: AutoFineTuneProgressState;
  message: string;
  provider_id?: 'hf-jobs';
  approval_required?: false;
  credential_readiness?: Record<string, boolean>;
  model_repo_url?: string;
  job_url?: string;
  eval_result?: string;
  estimated_total_cost_usd?: number;
  cost_cap_usd?: number;
  attempt?: number;
  max_attempts?: number;
  error_code?: string;
  updatedAt?: string;
}

export interface DatasetProfile {
  source?: Record<string, unknown>;
  format?: string | null;
  row_count?: number;
  profiled_row_count?: number;
  statistics_basis?: string;
  columns?: string[];
  inferred_shape?: string;
  missing_summary?: Record<string, number>;
  missing_fraction?: number;
  duplicate_count?: number;
  duplicate_fraction?: number;
  sample_rows?: Array<Record<string, unknown>>;
  trainability?: TrainabilitySummary;
  strategy?: StrategySummary;
  golden_eval?: GoldenEvalSummary;
  reference_lookup?: ReferenceLookupSummary;
  post_training_eval?: PostTrainingEvalSummary;
}

export interface DatasetUploadState {
  status: DatasetUploadStatus;
  repoId?: string;
  filename?: string;
  files?: Array<{
    filename?: string;
    format?: string;
    size_bytes?: number;
  }>;
  url?: string;
  uploadedAt?: string;
  error?: string;
  datasetProfile?: DatasetProfile;
  datasetProfileError?: string;
}

export interface SessionMeta {
  id: string;
  title: string;
  createdAt: string;
  isActive: boolean;
  needsAttention: boolean;
  vertical?: DomainId;
  provider?: CloudProviderId;
  dataset?: DatasetUploadState;
  domain_id?: DomainId;
  provider_id?: CloudProviderId;
  dataset_repo?: string | null;
  model?: string | null;
  autoApprovalEnabled?: boolean;
  autoApprovalCostCapUsd?: number | null;
  autoApprovalEstimatedSpendUsd?: number;
  autoApprovalRemainingUsd?: number | null;
  /** True when the backend no longer recognizes this session id (e.g.
   *  after a backend restart). The UI shows a recovery banner and
   *  disables input until the user chooses to restore-with-summary or
   *  start fresh. */
  expired?: boolean;
}

export interface ToolApproval {
  tool_call_id: string;
  approved: boolean;
  feedback?: string | null;
}

export interface User {
  authenticated: boolean;
  username?: string;
  name?: string;
  picture?: string;
  orgMember?: boolean;
}
