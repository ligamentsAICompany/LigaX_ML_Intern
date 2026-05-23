import { useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import {
  Box,
  Button,
  Chip,
  CircularProgress,
  Link,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import AnalyticsOutlinedIcon from '@mui/icons-material/AnalyticsOutlined';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import CloudQueueOutlinedIcon from '@mui/icons-material/CloudQueueOutlined';
import DatasetOutlinedIcon from '@mui/icons-material/DatasetOutlined';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import RadioButtonUncheckedIcon from '@mui/icons-material/RadioButtonUnchecked';
import ShieldOutlinedIcon from '@mui/icons-material/ShieldOutlined';
import TuneOutlinedIcon from '@mui/icons-material/TuneOutlined';
import type { ActivityStatus, ToolBudgetBlockState } from '@/store/agentStore';
import { useAgentStore } from '@/store/agentStore';
import { useSessionStore } from '@/store/sessionStore';
import type {
  AutoFineTuneProgress,
  CloudProviderId,
  DatasetProfile,
  DatasetUploadState,
  DomainId,
  GoldenEvalCase,
  GoldenEvalSummary,
  MLQualityRiskLevel,
  MLStrategyRecommendation,
  PostTrainingEvalSummary,
  ReferenceLookupSummary,
  StrategySummary,
  TrainabilitySummary,
} from '@/types/agent';
import { apiFetch } from '@/utils/api';
import { autoFineTuneResultFromOutput } from '@/utils/autoFineTuneResult';
import { PROVIDERS, type DomainTemplateOption, domainTemplateFromList, providerLabel, useDomainTemplates } from './domainOptions';

const cardSx = {
  border: '1px solid rgba(148,163,184,0.16)',
  borderRadius: '18px',
  background: 'linear-gradient(180deg, rgba(15,23,42,0.94), rgba(15,23,42,0.72))',
  p: { xs: 1.5, md: 1.75 },
  minWidth: 0,
  boxShadow: '0 18px 48px rgba(0,0,0,0.22)',
  backdropFilter: 'blur(16px)',
};

function money(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'unknown';
  if (value >= 100) return `$${value.toFixed(0)}`;
  return `$${value.toFixed(2).replace(/\.00$/, '')}`;
}

function firstBudgetBlock(blocks: Record<string, ToolBudgetBlockState>): ToolBudgetBlockState | null {
  return Object.values(blocks)[0] ?? null;
}

function statusCopy(status: ActivityStatus): { label: string; tone: 'idle' | 'active' | 'blocked' } {
  if (status.type === 'waiting-approval') return { label: 'Waiting for approval', tone: 'blocked' };
  if (status.type === 'tool') return { label: status.description || `Running ${status.toolName}`, tone: 'active' };
  if (status.type === 'thinking') return { label: 'Planning next step', tone: 'active' };
  if (status.type === 'streaming') return { label: 'Writing response', tone: 'active' };
  if (status.type === 'cancelled') return { label: 'Cancelled', tone: 'blocked' };
  return { label: 'Ready', tone: 'idle' };
}

function firstItems(items: string[], count = 2): string[] {
  return items.filter(Boolean).slice(0, count);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function asDatasetProfile(value: unknown): DatasetProfile | undefined {
  const record = asRecord(value);
  if (!record) return undefined;

  const profile: DatasetProfile = {};
  const source = asRecord(record.source);
  const missingSummary = asRecord(record.missing_summary);

  if (source) profile.source = source;
  if (typeof record.format === 'string' || record.format === null) profile.format = record.format;
  profile.row_count = asNumber(record.row_count);
  profile.profiled_row_count = asNumber(record.profiled_row_count);
  profile.statistics_basis = asString(record.statistics_basis);
  profile.columns = asStringArray(record.columns);
  profile.inferred_shape = asString(record.inferred_shape);
  if (missingSummary) profile.missing_summary = numericRecord(missingSummary);
  profile.missing_fraction = asNumber(record.missing_fraction);
  profile.duplicate_count = asNumber(record.duplicate_count);
  profile.duplicate_fraction = asNumber(record.duplicate_fraction);
  profile.sample_rows = asRecordArray(record.sample_rows);
  profile.trainability = asTrainability(record.trainability);
  profile.strategy = asStrategy(record.strategy);
  profile.golden_eval = asGoldenEval(record.golden_eval);
  profile.reference_lookup = asReferenceLookup(record.reference_lookup);
  profile.post_training_eval = asPostTrainingEval(record.post_training_eval);

  return profile;
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function asBoolean(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined;
}

function asStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const items = value.map(asString).filter((item): item is string => Boolean(item));
  return items.length ? items : undefined;
}

function asRecordArray(value: unknown): Array<Record<string, unknown>> | undefined {
  if (!Array.isArray(value)) return undefined;
  const items = value.map(asRecord).filter((item): item is Record<string, unknown> => Boolean(item));
  return items.length ? items : undefined;
}

function asUploadedFiles(value: unknown): DatasetUploadState['files'] {
  const records = asRecordArray(value);
  if (!records) return undefined;
  const files = records.map((record) => ({
    filename: asString(record.filename),
    format: asString(record.format),
    size_bytes: asNumber(record.size_bytes),
  }));
  return files.length ? files : undefined;
}

function numericRecord(value: Record<string, unknown>): Record<string, number> {
  return Object.fromEntries(
    Object.entries(value).filter((entry): entry is [string, number] => typeof entry[1] === 'number' && Number.isFinite(entry[1])),
  );
}

function asRiskLevel(value: unknown): MLQualityRiskLevel | undefined {
  return value === 'low' || value === 'medium' || value === 'high' ? value : undefined;
}

function asStrategyRecommendation(value: unknown): MLStrategyRecommendation | undefined {
  return value === 'fine_tune' || value === 'rag' || value === 'hybrid' || value === 'data_needed' ? value : undefined;
}

function asTrainability(value: unknown): TrainabilitySummary | undefined {
  const record = asRecord(value);
  if (!record) return undefined;
  return {
    score: asNumber(record.score),
    recommendation: asStrategyRecommendation(record.recommendation),
    risk_level: asRiskLevel(record.risk_level),
    reasons: asStringArray(record.reasons),
  };
}

function asStrategy(value: unknown): StrategySummary | undefined {
  const record = asRecord(value);
  if (!record) return undefined;
  return {
    strategy: asStrategyRecommendation(record.strategy),
    confidence: asNumber(record.confidence),
    risk_level: asRiskLevel(record.risk_level),
    reasons: asStringArray(record.reasons),
    required_next_actions: asStringArray(record.required_next_actions),
    can_train_without_override: asBoolean(record.can_train_without_override),
    requires_user_override_for_training: asBoolean(record.requires_user_override_for_training),
    method_hint: typeof record.method_hint === 'string' || record.method_hint === null ? record.method_hint : undefined,
    override_message: asString(record.override_message),
  };
}

function asGoldenEval(value: unknown): GoldenEvalSummary | undefined {
  const record = asRecord(value);
  if (!record) return undefined;
  const cases = asGoldenEvalCases(record.cases);
  const caseCount = asNumber(record.case_count) ?? cases?.length;
  return {
    case_count: caseCount,
    quality_constraints: asStringArray(record.quality_constraints),
    cases,
  };
}

function asGoldenEvalCases(value: unknown): GoldenEvalCase[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const cases = value
    .map(asRecord)
    .filter((item): item is Record<string, unknown> => Boolean(item))
    .map((item) => ({
      id: asString(item.id),
      task_type: asString(item.task_type),
    }));
  return cases.length ? cases : undefined;
}

function asReferenceLookup(value: unknown): ReferenceLookupSummary | undefined {
  const record = asRecord(value);
  if (!record) return undefined;
  return {
    ready: asBoolean(record.ready),
    row_count: asNumber(record.row_count),
    key_columns: asStringArray(record.key_columns),
    answer_columns: asStringArray(record.answer_columns),
    status: asString(record.status),
  };
}

function asPostTrainingEval(value: unknown): PostTrainingEvalSummary | undefined {
  const record = asRecord(value);
  if (!record) return undefined;
  const summary = asRecord(record.summary);
  return {
    status: asString(record.status),
    valid: asBoolean(record.valid),
    case_count: asNumber(record.case_count),
    passed: asNumber(record.passed),
    failed: asNumber(record.failed),
    needs_rag: asNumber(record.needs_rag),
    needs_more_data: asNumber(record.needs_more_data),
    top_reasons: asStringArray(record.top_reasons),
    summary: summary ? numericRecord(summary) : undefined,
  };
}

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'unknown';
  return `${Math.round(value * 100)}%`;
}

function formatConfidence(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'unknown';
  return `${Math.round(value * 100)}% confidence`;
}

function labelize(value: string | null | undefined): string {
  if (!value) return 'Unknown';
  return value.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

function firstReason(reasons: string[] | undefined, fallback: string): string {
  return firstItems(reasons ?? [], 1)[0] ?? fallback;
}

function postTrainingFromPanel(parameters: Record<string, unknown> | undefined): PostTrainingEvalSummary | undefined {
  const root = asRecord(parameters);
  const provenance = asRecord(root?.provenance) ?? root;
  const postTrainingEval = asRecord(provenance?.post_training_eval);
  if (!postTrainingEval) return undefined;
  return postTrainingEval as PostTrainingEvalSummary;
}

function autoFineTuneFromPanel(parameters: Record<string, unknown> | undefined): AutoFineTuneProgress | undefined {
  const root = asRecord(parameters);
  const progress = asRecord(root?.autoFineTune);
  if (!progress) return undefined;
  return progress as unknown as AutoFineTuneProgress;
}

interface SessionFlowPanelProps {
  sessionId: string;
}

export default function SessionFlowPanel({ sessionId }: SessionFlowPanelProps) {
  const { sessions, updateSessionVertical, updateSessionProvider, updateSessionDataset } = useSessionStore();
  const { sessionStates, activeSessionId, activityStatus, isProcessing, panelData, budgetBlocks } = useAgentStore();
  const domainTemplates = useDomainTemplates();
  const session = sessions.find((item) => item.id === sessionId);
  const activeState = sessionStates[sessionId];
  const effectiveActivity = activeSessionId === sessionId ? activityStatus : activeState?.activityStatus ?? { type: 'idle' as const };
  const effectivePanel = activeSessionId === sessionId ? panelData : activeState?.panelData ?? null;
  const effectiveProcessing = activeSessionId === sessionId ? isProcessing : activeState?.isProcessing ?? false;

  if (!session) return null;
  const vertical = session.vertical ?? session.domain_id ?? 'generic';
  const provider = session.provider ?? session.provider_id ?? 'hf-jobs';
  const selectedDomain = domainTemplateFromList(domainTemplates, vertical);
  const dataset = session.dataset ?? (
    session.dataset_repo
      ? {
          status: 'ready' as const,
          repoId: session.dataset_repo,
          url: `https://huggingface.co/datasets/${session.dataset_repo}`,
        }
      : { status: 'idle' as const }
  );
  const postTrainingQuality = dataset.datasetProfile?.post_training_eval ?? postTrainingFromPanel(effectivePanel?.parameters);
  const persistMetadata = (next: {
    domain_id?: DomainId;
    provider_id?: CloudProviderId;
    dataset_repo?: string | null;
  }) => {
    const datasetRepo = Object.prototype.hasOwnProperty.call(next, 'dataset_repo')
      ? next.dataset_repo
      : dataset.repoId ?? null;
    void apiFetch(`/api/session/${sessionId}/metadata`, {
      method: 'POST',
      body: JSON.stringify({
        domain_id: next.domain_id ?? vertical,
        provider_id: next.provider_id ?? provider,
        dataset_repo: datasetRepo,
      }),
    }).catch(() => {
      // The next chat send also carries metadata; this is best-effort reload recovery.
    });
  };
  const handleVerticalChange = (value: DomainId) => {
    updateSessionVertical(sessionId, value);
    persistMetadata({ domain_id: value });
  };
  const handleProviderChange = (value: CloudProviderId) => {
    updateSessionProvider(sessionId, value);
    persistMetadata({ provider_id: value });
  };
  const handleDatasetChange = (value: DatasetUploadState) => {
    updateSessionDataset(sessionId, value);
    persistMetadata({ dataset_repo: value.repoId ?? null });
  };
  const status = statusCopy(effectiveActivity);
  const hasDataset = dataset.status === 'ready';
  const hasProfile = Boolean(dataset.datasetProfile);
  const hasResults = Boolean(effectivePanel?.output || postTrainingQuality);
  const workflowSteps = [
    { label: 'Domain', caption: selectedDomain.label, done: true, active: false },
    { label: 'Dataset', caption: hasDataset ? 'Linked' : 'Awaiting source', done: hasDataset, active: !hasDataset },
    { label: 'Quality Intelligence', caption: hasProfile ? 'Profiled' : 'Not assessed', done: hasProfile, active: hasDataset && !hasProfile },
    { label: 'Provider/Cost', caption: providerLabel(provider), done: Boolean(provider), active: false },
    { label: 'Approval/Run', caption: status.label, done: effectiveActivity.type !== 'waiting-approval' && hasResults, active: effectiveProcessing || effectiveActivity.type === 'waiting-approval' },
    { label: 'Results', caption: hasResults ? 'Available' : 'Pending', done: hasResults, active: false },
  ];
  const progressFromPanel = autoFineTuneFromPanel(effectivePanel?.parameters);
  const resultFromOutput = autoFineTuneResultFromOutput(effectivePanel?.output?.content);
  const autoFineTuneProgress = progressFromPanel
    ? { ...progressFromPanel, ...resultFromOutput, model_repo_url: resultFromOutput.model_repo_url ?? progressFromPanel.model_repo_url }
    : resultFromOutput.model_repo_url || resultFromOutput.job_url
      ? {
          state: 'completed',
          message: 'Fine-tuned model is ready',
          provider_id: 'hf-jobs' as const,
          approval_required: false as const,
          ...resultFromOutput,
        }
      : undefined;

  const legacyCockpit = (
    <Box
      sx={{
        px: { xs: 0, md: 1 },
        pt: { xs: 0.5, md: 1 },
        flexShrink: 0,
      }}
    >
      <Box
        sx={{
          maxWidth: 1180,
          mx: 'auto',
          border: '1px solid rgba(148,163,184,0.18)',
          borderRadius: { xs: '18px', md: '26px' },
          background: 'linear-gradient(145deg, rgba(15,23,42,0.96), rgba(2,6,23,0.82))',
          p: { xs: 1.25, md: 1.75 },
          boxShadow: 'var(--shadow-1)',
          position: 'relative',
          overflow: 'hidden',
          maxHeight: { xs: '46vh', md: 'min(56vh, 620px)' },
          overflowY: 'auto',
          overscrollBehavior: 'contain',
          '&::-webkit-scrollbar': { width: 6 },
          '&::-webkit-scrollbar-thumb': {
            bgcolor: 'var(--scrollbar-thumb)',
            borderRadius: 999,
          },
          '&::before': {
            content: '""',
            position: 'absolute',
            inset: 0,
            pointerEvents: 'none',
            background: 'radial-gradient(circle at 0% 0%, rgba(34,197,94,0.16), transparent 32%), radial-gradient(circle at 100% 0%, rgba(56,189,248,0.08), transparent 30%)',
          },
        }}
      >
        <Stack spacing={1.5} sx={{ position: 'relative' }}>
          <Stack
            direction={{ xs: 'column', md: 'row' }}
            spacing={1.25}
            alignItems={{ xs: 'flex-start', md: 'center' }}
            justifyContent="space-between"
          >
            <Box sx={{ minWidth: 0 }}>
              <Typography sx={{ color: 'var(--accent-green)', fontSize: '0.68rem', fontWeight: 900, letterSpacing: '0.16em', textTransform: 'uppercase' }}>
                LigaX ML Intern
              </Typography>
              <Typography sx={{ color: 'var(--text)', fontSize: { xs: '1.08rem', md: '1.36rem' }, fontWeight: 900, letterSpacing: '-0.045em', lineHeight: 1.1, mt: 0.35 }}>
                Enterprise ML workflow cockpit
              </Typography>
              <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.76rem', lineHeight: 1.55, mt: 0.55, maxWidth: 720 }}>
                Configure the domain, attach data, review quality intelligence, choose cost controls, and launch only after approval gates are clear.
              </Typography>
            </Box>
            <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
              <Chip
                size="small"
                label={status.label}
                sx={{
                  height: 28,
                  fontWeight: 800,
                  bgcolor: status.tone === 'blocked' ? 'rgba(248,113,113,0.12)' : status.tone === 'active' ? 'var(--accent-green-weak)' : 'rgba(148,163,184,0.1)',
                  color: status.tone === 'blocked' ? 'var(--accent-red)' : status.tone === 'active' ? 'var(--accent-green)' : 'var(--muted-text)',
                  border: '1px solid rgba(148,163,184,0.18)',
                }}
              />
              <Chip
                size="small"
                label={providerLabel(provider)}
                sx={{ height: 28, fontWeight: 800, bgcolor: 'rgba(56,189,248,0.1)', color: 'var(--accent-blue)', border: '1px solid rgba(56,189,248,0.22)' }}
              />
            </Stack>
          </Stack>

          <WorkflowRail steps={workflowSteps} />

          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: {
                xs: '1fr',
                md: 'repeat(2, minmax(0, 1fr))',
                xl: '1.05fr 1.1fr 0.9fr 1.15fr',
              },
              gap: 1.25,
              alignItems: 'stretch',
            }}
        >
          <VerticalPicker
            value={vertical}
            options={domainTemplates}
            onChange={handleVerticalChange}
          />
          <DatasetPanel
            sessionId={sessionId}
            vertical={vertical}
            template={selectedDomain}
            dataset={dataset}
            onDatasetChange={handleDatasetChange}
          />
          <Box sx={{ display: 'grid', gridTemplateRows: 'auto auto', gap: 1.25, minWidth: 0 }}>
            <CloudPicker
              value={provider}
              onChange={handleProviderChange}
            />
            <CostPreview
              autoApprovalEnabled={Boolean(session.autoApprovalEnabled)}
              costCap={session.autoApprovalCostCapUsd}
              estimatedSpend={session.autoApprovalEstimatedSpendUsd}
              remaining={session.autoApprovalRemainingUsd}
              budgetBlock={firstBudgetBlock(budgetBlocks)}
            />
          </Box>
          <Box sx={{ display: 'grid', gridTemplateRows: 'auto auto', gap: 1.25, minWidth: 0 }}>
            <StarterKitPanel template={selectedDomain} />
            <JobTimeline
              activityStatus={effectiveActivity}
              isProcessing={effectiveProcessing}
              hasPanelData={Boolean(effectivePanel)}
            />
          </Box>
          <MLQualityPanel
            vertical={vertical}
            template={selectedDomain}
            dataset={dataset}
            panelTitle={effectivePanel?.title ?? null}
            hasOutput={Boolean(effectivePanel?.output)}
            postTrainingQuality={postTrainingQuality}
          />
          </Box>
        </Stack>
      </Box>
    </Box>
  );

  return (
    <>
      <AutoFineTunePanel
        sessionId={sessionId}
        vertical={vertical}
        template={selectedDomain}
        dataset={dataset}
        onDatasetChange={handleDatasetChange}
        activityStatus={effectiveActivity}
        isProcessing={effectiveProcessing}
        progress={autoFineTuneProgress}
      />
      <Box sx={{ display: 'none' }} aria-hidden>
        {legacyCockpit}
      </Box>
    </>
  );
}

function AutoFineTunePanel({
  sessionId,
  vertical,
  template,
  dataset,
  onDatasetChange,
  activityStatus,
  isProcessing,
  progress,
}: {
  sessionId: string;
  vertical: DomainId;
  template: DomainTemplateOption;
  dataset: DatasetUploadState;
  onDatasetChange: (dataset: DatasetUploadState) => void;
  activityStatus: ActivityStatus;
  isProcessing: boolean;
  progress?: AutoFineTuneProgress;
}) {
  const modelUrl = progress?.model_repo_url;
  const jobUrl = progress?.job_url;
  const evalResult = progress?.eval_result;
  const state = progress?.state;
  const failed = state === 'failed' || state === 'blocked';
  const status = progress?.message
    ?? (isProcessing || activityStatus.type === 'tool' ? statusCopy(activityStatus).label : 'Add a dataset, then send the prompt to start one clean run.');

  return (
    <Box sx={{ px: { xs: 0, md: 1 }, pt: { xs: 0.5, md: 1 }, flexShrink: 0 }}>
      <Box
        sx={{
          maxWidth: 1040,
          mx: 'auto',
          border: '1px solid rgba(148,163,184,0.18)',
          borderRadius: { xs: '18px', md: '28px' },
          background: 'linear-gradient(145deg, rgba(15,23,42,0.98), rgba(2,6,23,0.86))',
          boxShadow: '0 24px 90px rgba(0,0,0,0.42)',
          p: { xs: 1.35, md: 2 },
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            inset: 0,
            pointerEvents: 'none',
            background: 'radial-gradient(circle at 12% 0%, rgba(34,197,94,0.18), transparent 34%), radial-gradient(circle at 88% 12%, rgba(56,189,248,0.1), transparent 32%)',
          },
        }}
      >
        <Stack spacing={1.5} sx={{ position: 'relative' }}>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '0.95fr 1.35fr' }, gap: 1.25, alignItems: 'stretch' }}>
            <DatasetPanel
              sessionId={sessionId}
              vertical={vertical}
              template={template}
              dataset={dataset}
              onDatasetChange={onDatasetChange}
            />
            <Box sx={{ ...cardSx, minWidth: 0 }}>
              <SectionTitle kicker="Run command" label="Start the auto fine-tune" icon={<PlayCircleOutlineIcon sx={{ fontSize: 17 }} />} />
              <Stack spacing={1}>
                <Box sx={{ border: '1px solid rgba(34,197,94,0.2)', borderRadius: '14px', bgcolor: 'rgba(2,6,23,0.46)', px: 1.15, py: 1 }}>
                  <Typography sx={{ color: 'var(--text)', fontSize: '0.86rem', fontWeight: 850, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace' }}>
                    fine tune this dataset{dataset.repoId ? ` ${dataset.repoId}` : ''}
                  </Typography>
                </Box>
                <Typography sx={{ color: failed ? 'var(--accent-red)' : 'var(--muted-text)', fontSize: '0.72rem', lineHeight: 1.5 }}>
                  {status}
                </Typography>
              </Stack>
            </Box>
          </Box>

          {(modelUrl || jobUrl || evalResult || failed) && (
            <Box sx={{ ...cardSx, borderColor: failed ? 'rgba(248,113,113,0.34)' : 'rgba(34,197,94,0.28)' }}>
              <SectionTitle
                kicker={failed ? 'Run status' : 'Final result'}
                label={failed ? 'Auto fine-tune stopped' : 'Model ready'}
                icon={failed ? <ErrorOutlineIcon sx={{ fontSize: 17 }} /> : <CheckCircleOutlineIcon sx={{ fontSize: 17 }} />}
              />
              <Stack spacing={0.8}>
                {modelUrl ? (
                  <Link href={modelUrl} target="_blank" rel="noopener noreferrer" sx={{ color: 'var(--accent-green)', fontSize: '0.86rem', fontWeight: 900, wordBreak: 'break-all' }}>
                    {modelUrl}
                  </Link>
                ) : (
                  <Typography sx={{ color: failed ? 'var(--accent-red)' : 'var(--muted-text)', fontSize: '0.76rem' }}>
                    {progress?.message ?? 'Waiting for the backend to emit the Hugging Face model link.'}
                  </Typography>
                )}
                {jobUrl && (
                  <Link href={jobUrl} target="_blank" rel="noopener noreferrer" sx={{ color: 'var(--accent-blue)', fontSize: '0.72rem', fontWeight: 750, wordBreak: 'break-all' }}>
                    HF Job: {jobUrl}
                  </Link>
                )}
                {evalResult && (
                  <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.72rem', lineHeight: 1.45 }}>
                    Eval: {evalResult}
                  </Typography>
                )}
              </Stack>
            </Box>
          )}
        </Stack>
      </Box>
    </Box>
  );
}

function WorkflowRail({
  steps,
}: {
  steps: Array<{ label: string; caption: string; done: boolean; active: boolean }>;
}) {
  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: { xs: '1fr', md: 'repeat(6, minmax(0, 1fr))' },
        gap: 0.75,
        p: 0.75,
        border: '1px solid rgba(148,163,184,0.14)',
        borderRadius: '18px',
        bgcolor: 'rgba(2,6,23,0.42)',
      }}
      aria-label="Workflow progression"
    >
      {steps.map((step, index) => {
        const Icon = step.done ? CheckCircleOutlineIcon : step.active ? PlayCircleOutlineIcon : RadioButtonUncheckedIcon;
        return (
          <Stack
            key={step.label}
            direction="row"
            spacing={0.75}
            alignItems="center"
            sx={{
              minWidth: 0,
              p: 0.85,
              borderRadius: '13px',
              bgcolor: step.active ? 'var(--accent-green-weak)' : step.done ? 'rgba(34,197,94,0.07)' : 'transparent',
              border: '1px solid',
              borderColor: step.active ? 'rgba(34,197,94,0.32)' : 'transparent',
            }}
          >
            <Box
              sx={{
                width: 25,
                height: 25,
                flexShrink: 0,
                borderRadius: '10px',
                display: 'grid',
                placeItems: 'center',
                bgcolor: step.done ? 'rgba(34,197,94,0.16)' : step.active ? 'rgba(34,197,94,0.2)' : 'rgba(148,163,184,0.08)',
                color: step.done || step.active ? 'var(--accent-green)' : 'var(--muted-text)',
              }}
            >
              <Icon sx={{ fontSize: 16 }} />
            </Box>
            <Box sx={{ minWidth: 0 }}>
              <Typography sx={{ color: step.done || step.active ? 'var(--text)' : 'var(--muted-text)', fontSize: '0.68rem', fontWeight: 900, lineHeight: 1.15 }}>
                {index + 1}. {step.label}
              </Typography>
              <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.58rem', lineHeight: 1.25, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {step.caption}
              </Typography>
            </Box>
          </Stack>
        );
      })}
    </Box>
  );
}

function SectionTitle({ label, kicker, icon }: { label: string; kicker: string; icon?: ReactNode }) {
  return (
    <Box sx={{ mb: 1.15, display: 'flex', alignItems: 'flex-start', gap: 0.85 }}>
      {icon && (
        <Box
          sx={{
            width: 30,
            height: 30,
            borderRadius: '11px',
            display: 'grid',
            placeItems: 'center',
            color: 'var(--accent-green)',
            bgcolor: 'var(--accent-green-weak)',
            border: '1px solid rgba(34,197,94,0.18)',
            flexShrink: 0,
          }}
        >
          {icon}
        </Box>
      )}
      <Box sx={{ minWidth: 0 }}>
        <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.63rem', fontWeight: 900, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
          {kicker}
        </Typography>
        <Typography sx={{ color: 'var(--text)', fontSize: '0.9rem', fontWeight: 900, letterSpacing: '-0.02em', lineHeight: 1.2 }}>
          {label}
        </Typography>
      </Box>
    </Box>
  );
}

function VerticalPicker({
  value,
  options,
  onChange,
}: {
  value: DomainId;
  options: DomainTemplateOption[];
  onChange: (value: DomainId) => void;
}) {
  return (
    <Box sx={{ ...cardSx, minWidth: 0 }}>
      <SectionTitle kicker="Domain" label={domainTemplateFromList(options, value).label} icon={<TuneOutlinedIcon sx={{ fontSize: 17 }} />} />
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', sm: 'repeat(auto-fit, minmax(118px, 1fr))', xl: '1fr' }, gap: 0.75 }}>
        {options.map((option) => {
          const selected = option.id === value;
          return (
            <Box
              key={option.id}
              component="button"
              type="button"
              onClick={() => onChange(option.id)}
              sx={{
                textAlign: 'left',
                border: '1px solid',
                borderColor: selected ? 'rgba(34,197,94,0.55)' : 'var(--border)',
                borderRadius: '10px',
                bgcolor: selected ? 'var(--accent-green-weak)' : 'rgba(2,6,23,0.22)',
                color: 'var(--text)',
                cursor: 'pointer',
                p: 0.9,
                minHeight: 58,
                transition: 'border-color 0.16s ease, background-color 0.16s ease, transform 0.16s ease',
                '&:hover': { borderColor: 'var(--accent-green)', transform: 'translateY(-1px)' },
                '&:focus-visible': { outline: 'none', boxShadow: 'var(--focus)' },
              }}
            >
              <Typography sx={{ fontSize: '0.76rem', fontWeight: 750, lineHeight: 1.15 }}>
                {option.label}
              </Typography>
              <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.64rem', lineHeight: 1.25, mt: 0.35 }}>
                {option.detail}
              </Typography>
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}

function DatasetPanel({
  sessionId,
  vertical,
  template,
  dataset,
  onDatasetChange,
}: {
  sessionId: string;
  vertical: DomainId;
  template: DomainTemplateOption;
  dataset: DatasetUploadState;
  onDatasetChange: (dataset: DatasetUploadState) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const user = useAgentStore((s) => s.user);
  const defaultRepo = useMemo(() => {
    const namespace = user?.username || 'ml-intern';
    return `${namespace}/${vertical}-session-${sessionId.slice(0, 8)}`;
  }, [sessionId, user?.username, vertical]);
  const [repoId, setRepoId] = useState(dataset.repoId || defaultRepo);
  const datasetHint =
    template.starterKit?.datasetGuidance[0]
    ?? template.suggestedDatasets[0]
    ?? 'Uses the existing dataset upload API. You can also describe a private dataset in chat.';

  function updateRepo(value: string) {
    setRepoId(value);
    const trimmed = value.trim();
    onDatasetChange(
      trimmed
        ? {
            status: 'ready',
            repoId: trimmed,
            url: `https://huggingface.co/datasets/${trimmed}`,
          }
        : { status: 'idle' },
    );
  }

  async function upload(files: File[]) {
    if (!files.length) return;
    const targetRepo = repoId.trim() || defaultRepo;
    const uploadLabel = files.length === 1 ? files[0].name : `${files.length} files`;
    onDatasetChange({ status: 'uploading', repoId: targetRepo, filename: uploadLabel });
    const body = new FormData();
    body.append('repo_id', targetRepo);
    files.forEach((file) => body.append('files', file));
    try {
      const response = await apiFetch('/api/platform/upload-dataset', {
        method: 'POST',
        body,
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || 'Upload failed');
      }
      const payload = await response.json();
      onDatasetChange({
        status: 'ready',
        repoId: String(payload.dataset_id || targetRepo),
        filename: String(payload.filename || uploadLabel),
        files: asUploadedFiles(payload.files),
        url: String(payload.url || ''),
        uploadedAt: new Date().toISOString(),
        datasetProfile: asDatasetProfile(payload.dataset_profile),
        datasetProfileError: asString(payload.dataset_profile_error),
      });
    } catch (error) {
      onDatasetChange({
        status: 'error',
        repoId: targetRepo,
        filename: uploadLabel,
        error: error instanceof Error ? error.message.slice(0, 160) : 'Upload failed',
      });
    }
  }

  return (
    <Box sx={{ ...cardSx, minWidth: 0 }}>
      <SectionTitle kicker="Dataset" label={dataset.status === 'ready' ? 'Hub dataset linked' : 'Upload or describe'} icon={<DatasetOutlinedIcon sx={{ fontSize: 17 }} />} />
      <Stack spacing={0.85}>
        <TextField
          size="small"
          value={repoId}
          onChange={(event) => updateRepo(event.target.value)}
          label="Dataset repo"
          aria-label="Dataset repository ID"
          sx={{ '& .MuiInputBase-input': { fontSize: '0.74rem', fontWeight: 650 } }}
        />
        <input
          ref={fileInputRef}
          type="file"
          hidden
          multiple
          accept=".pdf,.docx,.csv,.xlsx"
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            if (files.length) void upload(files);
            event.target.value = '';
          }}
        />
        <Button
          variant="outlined"
          disabled={dataset.status === 'uploading'}
          onClick={() => fileInputRef.current?.click()}
          sx={{
            justifyContent: 'space-between',
            fontSize: '0.76rem',
            borderColor: 'rgba(34,197,94,0.28)',
            color: 'var(--text)',
            bgcolor: 'rgba(34,197,94,0.06)',
            '&:hover': { borderColor: 'var(--accent-green)', bgcolor: 'rgba(34,197,94,0.1)' },
          }}
        >
          {dataset.status === 'uploading' ? 'Uploading dataset' : 'Choose dataset files'}
          {dataset.status === 'uploading' && <CircularProgress size={14} />}
        </Button>
        <Typography sx={{ color: dataset.status === 'error' ? 'var(--accent-red)' : 'var(--muted-text)', fontSize: '0.68rem', lineHeight: 1.45 }}>
          {dataset.status === 'ready' ? (
            <>
              {dataset.files?.length ? `${dataset.files.length} source files -> ${dataset.filename}` : dataset.filename} ·{' '}
              {dataset.url ? (
                <Link href={dataset.url} target="_blank" rel="noopener noreferrer" sx={{ color: 'var(--accent-green)' }}>
                  Open on Hub
                </Link>
              ) : (
                dataset.repoId
              )}
            </>
          ) : dataset.status === 'error' ? (
            dataset.error || 'Upload failed.'
          ) : (
            datasetHint
          )}
        </Typography>
      </Stack>
    </Box>
  );
}

function StarterKitPanel({ template }: { template: DomainTemplateOption }) {
  const starterKit = template.starterKit;
  const prompts = firstItems(starterKit?.starterPrompts ?? template.contextInstructions);
  const models = firstItems(starterKit?.recommendedBaseModels ?? [], 2);
  const columns = firstItems(starterKit?.expectedColumns ?? [], 3);
  const labels = firstItems(starterKit?.expectedLabels ?? [], 3);

  return (
    <Box sx={{ ...cardSx, minWidth: 0 }}>
      <SectionTitle kicker="Starter kit" label={starterKit ? template.label : 'Generic guidance'} icon={<ShieldOutlinedIcon sx={{ fontSize: 17 }} />} />
      <Stack spacing={0.75}>
        {prompts.map((prompt) => (
          <Typography key={prompt} sx={{ color: 'var(--text)', fontSize: '0.68rem', lineHeight: 1.45 }}>
            {prompt}
          </Typography>
        ))}
        {models.length > 0 && (
          <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.64rem', lineHeight: 1.45 }}>
            Base models: {models.join(', ')}
          </Typography>
        )}
        {(columns.length > 0 || labels.length > 0) && (
          <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.64rem', lineHeight: 1.45 }}>
            Expected {columns.length > 0 ? `columns: ${columns.join(', ')}` : ''}
            {columns.length > 0 && labels.length > 0 ? ' · ' : ''}
            {labels.length > 0 ? `labels: ${labels.join(', ')}` : ''}
          </Typography>
        )}
      </Stack>
    </Box>
  );
}

function CloudPicker({ value, onChange }: { value: CloudProviderId; onChange: (value: CloudProviderId) => void }) {
  return (
    <Box sx={{ ...cardSx, minWidth: 0 }}>
      <SectionTitle kicker="Provider" label="Cloud runtime" icon={<CloudQueueOutlinedIcon sx={{ fontSize: 17 }} />} />
      <Stack spacing={0.65}>
        {PROVIDERS.map((provider) => {
          const selected = provider.id === value;
          return (
            <Box
              key={provider.id}
              component="button"
              type="button"
              disabled={!provider.enabled}
              onClick={() => provider.enabled && onChange(provider.id)}
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 1,
                border: '1px solid',
                borderColor: selected ? 'rgba(34,197,94,0.5)' : 'var(--border)',
                borderRadius: '10px',
                bgcolor: selected ? 'var(--accent-green-weak)' : 'rgba(2,6,23,0.22)',
                color: provider.enabled ? 'var(--text)' : 'var(--muted-text)',
                opacity: provider.enabled ? 1 : 0.55,
                cursor: provider.enabled ? 'pointer' : 'not-allowed',
                p: 0.85,
                textAlign: 'left',
                '&:hover': provider.enabled ? { borderColor: 'var(--accent-green)' } : undefined,
                '&:focus-visible': { outline: 'none', boxShadow: 'var(--focus)' },
              }}
            >
              <Box>
                <Typography sx={{ fontSize: '0.74rem', fontWeight: 750 }}>{provider.label}</Typography>
                <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.62rem' }}>{provider.detail}</Typography>
              </Box>
              <Chip
                size="small"
                label={provider.id === 'hf-jobs' ? 'live' : provider.enabled ? 'plan' : 'soon'}
                sx={{ height: 18, fontSize: '0.58rem', bgcolor: provider.enabled ? 'var(--accent-green-weak)' : 'var(--hover-bg)', color: provider.enabled ? 'var(--accent-green)' : 'var(--muted-text)' }}
              />
            </Box>
          );
        })}
      </Stack>
    </Box>
  );
}

function CostPreview({
  autoApprovalEnabled,
  costCap,
  estimatedSpend,
  remaining,
  budgetBlock,
}: {
  autoApprovalEnabled: boolean;
  costCap?: number | null;
  estimatedSpend?: number;
  remaining?: number | null;
  budgetBlock: ToolBudgetBlockState | null;
}) {
  return (
    <Box sx={{ ...cardSx, minWidth: 0 }}>
      <SectionTitle kicker="Cost" label={budgetBlock ? 'Approval needed' : 'Guardrails'} icon={<AnalyticsOutlinedIcon sx={{ fontSize: 17 }} />} />
      <Stack spacing={0.75}>
        <Typography sx={{ color: budgetBlock ? 'var(--accent-yellow)' : 'var(--text)', fontSize: '0.88rem', fontWeight: 900 }}>
          {budgetBlock ? money(budgetBlock.estimatedCostUsd) : autoApprovalEnabled ? `${money(remaining)} remaining` : 'Manual approval'}
        </Typography>
        <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.68rem', lineHeight: 1.5 }}>
          {budgetBlock
            ? `Remaining cap: ${money(budgetBlock.remainingCapUsd)}${budgetBlock.reason ? `. ${budgetBlock.reason}` : ''}`
            : autoApprovalEnabled
              ? `Spend ${money(estimatedSpend ?? 0)} of ${money(costCap)}. Scheduled and unknown-cost jobs still ask first.`
              : 'Billable or unknown-cost work stays behind the existing approval flow.'}
        </Typography>
      </Stack>
    </Box>
  );
}

function JobTimeline({
  activityStatus,
  isProcessing,
  hasPanelData,
}: {
  activityStatus: ActivityStatus;
  isProcessing: boolean;
  hasPanelData: boolean;
}) {
  const copy = statusCopy(activityStatus);
  const steps = [
    { label: 'Session', done: true },
    { label: 'Approval', done: activityStatus.type !== 'waiting-approval', active: activityStatus.type === 'waiting-approval' },
    { label: 'Run', done: hasPanelData && !isProcessing, active: isProcessing },
  ];
  return (
    <Box sx={{ ...cardSx, minWidth: 0 }}>
      <SectionTitle kicker="Approval / Run" label={copy.label} icon={<PlayCircleOutlineIcon sx={{ fontSize: 17 }} />} />
      <Stack spacing={0.65}>
        {steps.map((step) => (
          <Stack key={step.label} direction="row" spacing={0.75} alignItems="center">
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                bgcolor: step.active ? 'var(--accent-green)' : step.done ? 'var(--accent-green)' : 'var(--border-hover)',
                boxShadow: step.active ? '0 0 0 4px rgba(34,197,94,0.14)' : 'none',
                flexShrink: 0,
              }}
            />
            <Typography sx={{ color: step.active ? 'var(--text)' : 'var(--muted-text)', fontSize: '0.68rem', fontWeight: step.active ? 700 : 500 }}>
              {step.label}
            </Typography>
          </Stack>
        ))}
        <Chip
          size="small"
          label={copy.tone === 'blocked' ? 'action needed' : copy.tone === 'active' ? 'live' : 'idle'}
          sx={{ alignSelf: 'flex-start', height: 20, fontSize: '0.62rem', bgcolor: copy.tone === 'blocked' ? 'rgba(248,113,113,0.12)' : copy.tone === 'active' ? 'var(--accent-green-weak)' : 'var(--hover-bg)', color: copy.tone === 'blocked' ? 'var(--accent-red)' : copy.tone === 'active' ? 'var(--accent-green)' : 'var(--muted-text)' }}
        />
      </Stack>
    </Box>
  );
}

function MetricLine({ label, value, tone = 'muted' }: { label: string; value: string; tone?: 'muted' | 'good' | 'warn' | 'risk' }) {
  const color = tone === 'good'
    ? 'var(--accent-green)'
    : tone === 'warn'
      ? 'var(--accent-yellow)'
      : tone === 'risk'
        ? 'var(--accent-red)'
        : 'var(--muted-text)';
  return (
    <Stack direction="row" spacing={1} justifyContent="space-between" alignItems="baseline">
      <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.63rem' }}>{label}</Typography>
      <Typography sx={{ color, fontSize: '0.66rem', fontWeight: 750, textAlign: 'right' }}>{value}</Typography>
    </Stack>
  );
}

function QualityChip({ label, tone = 'muted' }: { label: string; tone?: 'muted' | 'good' | 'warn' | 'risk' }) {
  const palette = {
    muted: { bgcolor: 'var(--hover-bg)', color: 'var(--muted-text)' },
    good: { bgcolor: 'rgba(47,204,113,0.12)', color: 'var(--accent-green)' },
    warn: { bgcolor: 'var(--accent-yellow-weak)', color: 'var(--accent-yellow)' },
    risk: { bgcolor: 'rgba(224,90,79,0.12)', color: 'var(--accent-red)' },
  }[tone];
  return <Chip size="small" label={label} sx={{ height: 22, fontSize: '0.6rem', fontWeight: 850, border: '1px solid rgba(148,163,184,0.14)', ...palette }} />;
}

function MiniPanel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Box sx={{ border: '1px solid rgba(148,163,184,0.14)', borderRadius: '14px', p: 1.05, bgcolor: 'rgba(2,6,23,0.34)', minWidth: 0 }}>
      <Typography sx={{ color: 'var(--text)', fontSize: '0.7rem', fontWeight: 900, mb: 0.65, letterSpacing: '-0.01em' }}>
        {title}
      </Typography>
      {children}
    </Box>
  );
}

function MLQualityPanel({
  vertical,
  template,
  dataset,
  panelTitle,
  hasOutput,
  postTrainingQuality,
}: {
  vertical: DomainId;
  template: DomainTemplateOption;
  dataset: DatasetUploadState;
  panelTitle: string | null;
  hasOutput: boolean;
  postTrainingQuality?: PostTrainingEvalSummary;
}) {
  const profile = dataset.datasetProfile;
  const trainability = profile?.trainability;
  const strategy = profile?.strategy;
  const goldenEval = profile?.golden_eval;
  const referenceLookup = profile?.reference_lookup;
  const evaluationHint = template.starterKit?.evaluationRubric[0] ?? template.evaluationHints[0] ?? template.placeholderPrompt;
  const complianceHint = template.starterKit?.complianceNotes[0] ?? template.complianceNotes[0];
  const risk = trainability?.risk_level;
  const riskTone = risk === 'low' ? 'good' : risk === 'medium' ? 'warn' : risk === 'high' ? 'risk' : 'muted';
  const profileFormat = String(profile?.format ?? profile?.source?.format ?? '').toLowerCase();
  const structuredReference = profile?.inferred_shape === 'structured_reference_table' || referenceLookup?.ready === true;
  const incomeTaxReferenceTable = vertical === 'itr' && (
    structuredReference || ['xlsx', 'csv'].includes(profileFormat)
  );
  const strategyRisk = strategy?.risk_level ?? risk;
  const directFineTuneRisk = risk === 'high' || strategy?.requires_user_override_for_training === true || (
    incomeTaxReferenceTable && strategy?.strategy !== 'fine_tune'
  );
  const directFineTuneSafe = strategy?.strategy === 'fine_tune'
    && strategy.can_train_without_override === true
    && (strategyRisk === 'low' || strategyRisk === 'medium')
    && !directFineTuneRisk;
  const directFineTuneStatus = directFineTuneSafe ? 'good' : directFineTuneRisk ? 'risk' : profile ? 'warn' : 'muted';
  const directFineTuneLabel = directFineTuneSafe
    ? 'Direct fine-tune safe'
    : directFineTuneRisk
      ? 'Direct fine-tune blocked'
      : 'Needs backend assessment';
  const goldenCaseCount = goldenEval?.case_count ?? 0;
  const goldenTaskTypes = firstItems([
    ...new Set((goldenEval?.cases ?? []).map((item) => item.task_type).filter((item): item is string => Boolean(item))),
  ], 3);
  const readinessItems = [
    { label: 'Dataset profiled', status: profile ? 'good' : 'muted' },
    { label: directFineTuneLabel, status: directFineTuneStatus },
    { label: 'Golden eval ready', status: goldenCaseCount > 0 ? 'good' : 'muted' },
    { label: 'RAG/reference path', status: referenceLookup?.ready || strategy?.strategy === 'rag' || strategy?.strategy === 'hybrid' ? 'good' : 'muted' },
  ];
  const postSummary = postTrainingQuality?.summary;
  const postFailed = postTrainingQuality?.failed ?? postSummary?.failed;
  const postNeedsRag = postTrainingQuality?.needs_rag ?? postSummary?.needs_rag;

  return (
    <Box sx={{ ...cardSx, minWidth: 0, gridColumn: { xs: 'auto', md: '1 / -1', xl: 'span 4' } }}>
      <SectionTitle
        kicker="Quality Intelligence"
        label={profile ? `${labelize(strategy?.strategy)} recommendation` : 'Awaiting dataset profile'}
        icon={<AnalyticsOutlinedIcon sx={{ fontSize: 17 }} />}
      />
      <Stack spacing={0.85}>
        {directFineTuneRisk && (
          <Box sx={{ border: '1px solid rgba(248,113,113,0.38)', borderRadius: '14px', p: 1, bgcolor: 'rgba(248,113,113,0.1)', display: 'flex', gap: 1, alignItems: 'flex-start' }}>
            <ErrorOutlineIcon sx={{ color: 'var(--accent-red)', fontSize: 18, mt: 0.1, flexShrink: 0 }} />
            <Box>
            <Typography sx={{ color: 'var(--accent-red)', fontSize: '0.72rem', fontWeight: 900 }}>
              High-risk direct fine-tune
            </Typography>
            <Typography sx={{ color: 'var(--text)', fontSize: '0.64rem', lineHeight: 1.45, mt: 0.35 }}>
              {incomeTaxReferenceTable
                ? 'Income-tax reference-table data should use RAG, hybrid grounding, or deterministic lookup before any override-driven fine-tune.'
                : strategy?.override_message || firstReason(trainability?.reasons, 'Quality gates require review before training.')}
            </Typography>
            </Box>
          </Box>
        )}
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(auto-fit, minmax(min(100%, 235px), 1fr))' }, gap: 0.9, minWidth: 0 }}>
          <MiniPanel title="Dataset Health">
            <Stack spacing={0.45}>
              <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                <QualityChip label={risk ? `${labelize(risk)} risk` : 'No profile'} tone={riskTone} />
                <QualityChip label={profile?.inferred_shape ? labelize(profile.inferred_shape) : dataset.status} tone={profile ? 'good' : 'muted'} />
              </Stack>
              <MetricLine label="Rows" value={profile?.row_count?.toLocaleString() ?? 'unknown'} />
              <MetricLine label="Columns" value={profile?.columns?.length.toString() ?? 'unknown'} />
              <MetricLine label="Missing" value={formatPercent(profile?.missing_fraction)} tone={(profile?.missing_fraction ?? 0) >= 0.1 ? 'warn' : 'muted'} />
              <MetricLine label="Duplicates" value={formatPercent(profile?.duplicate_fraction)} tone={(profile?.duplicate_fraction ?? 0) >= 0.1 ? 'warn' : 'muted'} />
              {dataset.datasetProfileError && (
                <Typography sx={{ color: 'var(--accent-yellow)', fontSize: '0.62rem', lineHeight: 1.35 }}>
                  {dataset.datasetProfileError}
                </Typography>
              )}
            </Stack>
          </MiniPanel>
          <MiniPanel title="Strategy Recommendation">
            <Stack spacing={0.5}>
              <MetricLine label="Strategy" value={labelize(strategy?.strategy ?? trainability?.recommendation)} tone={directFineTuneRisk ? 'risk' : directFineTuneSafe ? 'good' : 'warn'} />
              <MetricLine label="Confidence" value={formatConfidence(strategy?.confidence)} />
              <MetricLine label="Trainability score" value={trainability?.score !== undefined ? `${trainability.score}/100` : 'unknown'} tone={riskTone} />
              <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.62rem', lineHeight: 1.4 }}>
                {firstReason(strategy?.reasons ?? trainability?.reasons, evaluationHint)}
              </Typography>
            </Stack>
          </MiniPanel>
          <MiniPanel title="Training Readiness">
            <Stack spacing={0.45}>
              {readinessItems.map((item) => (
                <Stack key={item.label} direction="row" spacing={0.65} alignItems="center">
                  <Box sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    bgcolor: item.status === 'risk'
                      ? 'var(--accent-red)'
                      : item.status === 'warn'
                        ? 'var(--accent-yellow)'
                        : item.status === 'good'
                          ? 'var(--accent-green)'
                          : 'var(--border-hover)',
                  }} />
                  <Typography sx={{
                    color: item.status === 'risk'
                      ? 'var(--accent-red)'
                      : item.status === 'warn'
                        ? 'var(--accent-yellow)'
                        : item.status === 'good'
                          ? 'var(--text)'
                          : 'var(--muted-text)',
                    fontSize: '0.64rem',
                    fontWeight: item.status === 'risk' || item.status === 'warn' ? 800 : 600,
                  }}>
                    {item.label}
                  </Typography>
                </Stack>
              ))}
              {firstItems(strategy?.required_next_actions ?? [], 1).map((action) => (
                <Typography key={action} sx={{ color: 'var(--muted-text)', fontSize: '0.62rem', lineHeight: 1.4 }}>
                  Next: {action}
                </Typography>
              ))}
            </Stack>
          </MiniPanel>
          <MiniPanel title="Golden Eval Preview">
            <Stack spacing={0.45}>
              <MetricLine label="Cases" value={String(goldenCaseCount)} tone={goldenCaseCount > 0 ? 'good' : 'muted'} />
              <MetricLine label="Task types" value={goldenTaskTypes.join(', ') || 'unknown'} />
              <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.62rem', lineHeight: 1.35 }}>
                {goldenCaseCount > 0
                  ? 'Eval cases are available. Raw questions and expected answers are hidden from this preview.'
                  : 'Upload or inspect a dataset to generate source-grounded eval cases.'}
              </Typography>
            </Stack>
          </MiniPanel>
          <MiniPanel title="RAG / Reference Lookup">
            <Stack spacing={0.45}>
              <MetricLine label="Readiness" value={referenceLookup?.ready ? 'Ready' : profile ? labelize(referenceLookup?.status) : 'Awaiting profile'} tone={referenceLookup?.ready ? 'good' : incomeTaxReferenceTable ? 'warn' : 'muted'} />
              <MetricLine label="Indexed rows" value={referenceLookup?.row_count?.toLocaleString() ?? 'unknown'} />
              <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.62rem', lineHeight: 1.4 }}>
                {referenceLookup?.ready
                  ? `Keys: ${firstItems(referenceLookup.key_columns ?? [], 2).join(', ') || 'detected'} · Answers: ${firstItems(referenceLookup.answer_columns ?? [], 2).join(', ') || 'detected'}`
                  : incomeTaxReferenceTable
                    ? 'Reference lookup is the preferred readiness path for structured tax tables.'
                    : 'RAG readiness appears when structured key and answer columns are detected.'}
              </Typography>
            </Stack>
          </MiniPanel>
          <MiniPanel title="Post-Training Quality">
            <Stack spacing={0.45}>
              <MetricLine label="Status" value={postTrainingQuality?.status ? labelize(postTrainingQuality.status) : 'Not run'} tone={postTrainingQuality?.status === 'passed' ? 'good' : postTrainingQuality ? 'warn' : 'muted'} />
              <MetricLine label="Cases" value={postTrainingQuality?.case_count?.toString() ?? 'unknown'} />
              <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.62rem', lineHeight: 1.4 }}>
                {postTrainingQuality
                  ? `Failures: ${postFailed ?? 0} · Needs RAG: ${postNeedsRag ?? 0}`
                  : hasOutput
                    ? `${panelTitle || 'Output'} is available; structured post-training eval appears here when emitted.`
                    : 'No model quality report exists yet.'}
              </Typography>
            </Stack>
          </MiniPanel>
        </Box>
      {complianceHint && (
        <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.64rem', lineHeight: 1.45, mt: 0.75 }}>
          Compliance: {complianceHint}
        </Typography>
      )}
      </Stack>
    </Box>
  );
}

