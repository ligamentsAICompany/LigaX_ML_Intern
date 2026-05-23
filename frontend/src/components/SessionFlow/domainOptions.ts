import { useEffect, useState } from 'react';
import type { CloudProviderId, DomainId } from '@/types/agent';
import { apiFetch } from '@/utils/api';

export interface StarterKitOption {
  starterKitId: DomainId;
  starterPrompts: string[];
  datasetGuidance: string[];
  recommendedDatasets: string[];
  recommendedBaseModels: string[];
  evaluationRubric: string[];
  metrics: string[];
  expectedColumns: string[];
  expectedLabels: string[];
  complianceNotes: string[];
  workflowSteps: string[];
}

export interface DomainTemplateOption {
  id: DomainId;
  label: string;
  detail: string;
  placeholderPrompt: string;
  contextInstructions: string[];
  suggestedDatasets: string[];
  evaluationHints: string[];
  complianceNotes: string[];
  fastPathInstructions?: string[];
  starterKit?: StarterKitOption | null;
}

export const DOMAIN_OPTIONS: DomainTemplateOption[] = [
  {
    id: 'generic',
    label: 'Generic',
    detail: 'Open-ended ML workflow',
    placeholderPrompt: 'Describe the dataset, target metric, and model goal.',
    contextInstructions: [
      'Use the selected workflow context unless the user explicitly asks to change it.',
      'Clarify missing dataset, metric, label, and deployment constraints before proposing spend.',
    ],
    suggestedDatasets: [],
    evaluationHints: ['Choose task-appropriate metrics and explain baseline expectations.'],
    complianceNotes: ['Do not expose secrets, tokens, private repo contents, or personal data.'],
  },
  {
    id: 'itr',
    label: 'ITR',
    detail: 'Tax forms and income signals',
    placeholderPrompt: 'Upload or describe ITR-style records, labels, and validation rules.',
    contextInstructions: [
      'Treat the task as income-tax return document or structured tax-record modeling.',
      'Ask for schema, assessment year, target label, and validation rules when missing.',
    ],
    suggestedDatasets: ['Customer-provided private ITR dataset'],
    evaluationHints: ['Report field-level extraction quality, validation-rule pass rates, and task-specific accuracy.'],
    complianceNotes: ['Redact PAN, Aadhaar, bank account numbers, phone numbers, addresses, and tax identifiers.'],
    starterKit: {
      starterKitId: 'itr',
      starterPrompts: ['Extract income, deduction, and filing-status signals from private ITR records.'],
      datasetGuidance: ['Bring a private, permissioned ITR dataset with clear labels and redaction status.'],
      recommendedDatasets: ['Customer-provided private ITR dataset'],
      recommendedBaseModels: ['Qwen/Qwen2.5-1.5B-Instruct', 'microsoft/Phi-3.5-mini-instruct'],
      evaluationRubric: ['Preserve numeric fields, assessment-year references, and validation-rule evidence.'],
      metrics: ['field-level F1', 'validation-rule pass rate'],
      expectedColumns: ['assessment_year', 'income_fields', 'deduction_fields', 'target_label'],
      expectedLabels: ['filing_status', 'review_required'],
      complianceNotes: ['Redact PAN, Aadhaar, bank account numbers, addresses, and tax identifiers.'],
      workflowSteps: ['Confirm schema and target label.', 'Run redaction checks before training.'],
    },
  },
  {
    id: 'gst',
    label: 'GST',
    detail: 'Invoice and compliance datasets',
    placeholderPrompt: 'Frame the GST prediction, anomaly, or extraction task.',
    contextInstructions: [
      'Treat the task as GST invoice, filing, anomaly detection, or compliance extraction work.',
      'Ask for invoice schema, tax periods, labels, and jurisdiction-specific rules when missing.',
    ],
    suggestedDatasets: ['Customer-provided private GST invoice or return dataset'],
    evaluationHints: ['Use extraction F1, anomaly precision/recall, reconciliation accuracy, or validation-rule pass rates.'],
    complianceNotes: ['Redact GSTINs, invoice numbers, addresses, bank details, phone numbers, and emails.'],
    starterKit: {
      starterKitId: 'gst',
      starterPrompts: ['Classify GST reconciliation status or extract invoice fields from private records.'],
      datasetGuidance: ['Use private GST invoices or returns with filing period, tax components, and labels.'],
      recommendedDatasets: ['Customer-provided private GST invoice or return dataset'],
      recommendedBaseModels: ['Qwen/Qwen2.5-1.5B-Instruct', 'google/gemma-2-2b-it'],
      evaluationRubric: ['Reconcile taxable value and tax components while separating mismatch reasons.'],
      metrics: ['extraction F1', 'anomaly precision/recall', 'reconciliation accuracy'],
      expectedColumns: ['filing_period', 'invoice_text', 'tax_components', 'target_label'],
      expectedLabels: ['reconciled', 'mismatch_reason', 'compliance_risk'],
      complianceNotes: ['Redact or hash GSTINs, invoice numbers, addresses, and bank details.'],
      workflowSteps: ['Confirm invoice schema and filing periods.', 'Validate private dataset location.'],
    },
  },
  {
    id: 'fieldops',
    label: 'FieldOps',
    detail: 'Inspections, tickets, and route work',
    placeholderPrompt: 'Summarize the field workflow, outcome label, and operational constraints.',
    contextInstructions: [
      'Treat the task as field operations, inspections, tickets, routing, or workforce support.',
      'Ask for workflow stage, SLA, location granularity, and target outcome when missing.',
    ],
    suggestedDatasets: ['Customer-provided field ticket, inspection, or route dataset'],
    evaluationHints: ['Track classification quality, SLA/routing accuracy, calibration, and operational error costs.'],
    complianceNotes: ['Redact customer names, addresses, phone numbers, GPS coordinates, and employee identifiers.'],
    starterKit: {
      starterKitId: 'fieldops',
      starterPrompts: ['Predict field ticket priority, SLA risk, or next best action from operations data.'],
      datasetGuidance: ['Use private tickets, inspections, routes, and technician notes with outcomes.'],
      recommendedDatasets: ['Customer-provided field ticket, inspection, or route dataset'],
      recommendedBaseModels: ['Qwen/Qwen2.5-1.5B-Instruct', 'microsoft/Phi-3.5-mini-instruct'],
      evaluationRubric: ['Preserve safety issues, blockers, and customer commitments.'],
      metrics: ['priority classification F1', 'SLA-risk precision/recall', 'routing accuracy'],
      expectedColumns: ['ticket_id', 'issue_description', 'technician_notes', 'sla_status', 'target_label'],
      expectedLabels: ['priority', 'next_best_action', 'sla_risk'],
      complianceNotes: ['Redact customer names, addresses, phone numbers, GPS coordinates, and employee identifiers.'],
      workflowSteps: ['Confirm operational workflow stage.', 'Define error costs before training.'],
    },
  },
  {
    id: 'call-center',
    label: 'Call Center',
    detail: 'Conversations, QA, and intent data',
    placeholderPrompt: 'Share the call transcript objective, labels, and evaluation rubric.',
    contextInstructions: [
      'Treat the task as support conversations, QA, intent routing, summarization, or agent-assist work.',
      'Ask for transcript format, labels, language mix, and quality rubric when missing.',
    ],
    suggestedDatasets: ['bitext/Bitext-customer-support-llm-chatbot-training-dataset'],
    evaluationHints: ['Use exact-match or F1 for labels, rubric pass rates for QA, and human-review samples for generation.'],
    complianceNotes: ['Redact names, emails, phone numbers, account numbers, addresses, and payment details.'],
    fastPathInstructions: ['Phase 6 approval-before-spend fast path'],
    starterKit: {
      starterKitId: 'call-center',
      starterPrompts: ['Fine-tune a support assistant for intent routing, QA, and safe escalation.'],
      datasetGuidance: ['Use transcripts with speaker roles, intents, QA labels, and redaction status.'],
      recommendedDatasets: ['bitext/Bitext-customer-support-llm-chatbot-training-dataset'],
      recommendedBaseModels: ['Qwen/Qwen2.5-1.5B-Instruct', 'google/gemma-2-2b-it'],
      evaluationRubric: ['Score correctness, empathy, policy adherence, escalation safety, and PII handling.'],
      metrics: ['intent accuracy', 'macro F1', 'rubric pass rate'],
      expectedColumns: ['conversation_id', 'turn_text', 'speaker_role', 'intent', 'target_response'],
      expectedLabels: ['intent', 'escalation_required', 'qa_score'],
      complianceNotes: ['Redact PII such as names, emails, phone numbers, account numbers, and payment details.'],
      workflowSteps: ['Confirm transcript format and labels.', 'Stop at approval_required before billable HF Jobs launch.'],
    },
  },
];

export const PROVIDERS: Array<{ id: CloudProviderId; label: string; detail: string; enabled: boolean }> = [
  { id: 'hf-jobs', label: 'HF Jobs', detail: 'Available now', enabled: true },
  { id: 'aws', label: 'AWS', detail: 'Plan-only preview', enabled: true },
  { id: 'azure', label: 'Azure', detail: 'Plan-only preview', enabled: true },
  { id: 'gcp', label: 'GCP', detail: 'Plan-only preview', enabled: true },
];

let cachedDomainOptions: DomainTemplateOption[] | null = null;
let pendingDomainOptions: Promise<DomainTemplateOption[]> | null = null;

function isDomainId(value: unknown): value is DomainId {
  return ['generic', 'itr', 'gst', 'fieldops', 'call-center'].includes(String(value));
}

function arrayOfStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function normalizeStarterKit(value: unknown): StarterKitOption | null {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Record<string, unknown>;
  const starterKitId = raw.starter_kit_id;
  if (!isDomainId(starterKitId)) return null;
  return {
    starterKitId,
    starterPrompts: arrayOfStrings(raw.starter_prompts),
    datasetGuidance: arrayOfStrings(raw.dataset_guidance),
    recommendedDatasets: arrayOfStrings(raw.recommended_datasets),
    recommendedBaseModels: arrayOfStrings(raw.recommended_base_models),
    evaluationRubric: arrayOfStrings(raw.evaluation_rubric),
    metrics: arrayOfStrings(raw.metrics),
    expectedColumns: arrayOfStrings(raw.expected_columns),
    expectedLabels: arrayOfStrings(raw.expected_labels),
    complianceNotes: arrayOfStrings(raw.compliance_notes),
    workflowSteps: arrayOfStrings(raw.workflow_steps),
  };
}

function normalizeTemplate(value: unknown): DomainTemplateOption | null {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Record<string, unknown>;
  if (!isDomainId(raw.domain_id)) return null;
  return {
    id: raw.domain_id,
    label: typeof raw.label === 'string' ? raw.label : raw.domain_id,
    detail: typeof raw.detail === 'string' ? raw.detail : '',
    placeholderPrompt: typeof raw.placeholder_prompt === 'string' ? raw.placeholder_prompt : DOMAIN_OPTIONS[0].placeholderPrompt,
    contextInstructions: arrayOfStrings(raw.context_instructions),
    suggestedDatasets: arrayOfStrings(raw.suggested_datasets),
    evaluationHints: arrayOfStrings(raw.evaluation_hints),
    complianceNotes: arrayOfStrings(raw.compliance_notes),
    fastPathInstructions: arrayOfStrings(raw.fast_path_instructions),
    starterKit: normalizeStarterKit(raw.starter_kit),
  };
}

export async function fetchDomainTemplates(): Promise<DomainTemplateOption[]> {
  if (cachedDomainOptions) return cachedDomainOptions;
  if (!pendingDomainOptions) {
    pendingDomainOptions = apiFetch('/api/domain-templates')
      .then(async (response) => {
        if (!response.ok) throw new Error('Domain templates unavailable');
        const payload = await response.json();
        const templates = Array.isArray(payload)
          ? payload.map(normalizeTemplate).filter((item): item is DomainTemplateOption => item !== null)
          : [];
        if (!templates.length) throw new Error('No domain templates returned');
        cachedDomainOptions = templates;
        return templates;
      })
      .catch(() => DOMAIN_OPTIONS)
      .finally(() => {
        pendingDomainOptions = null;
      });
  }
  return pendingDomainOptions;
}

export function useDomainTemplates(): DomainTemplateOption[] {
  const [templates, setTemplates] = useState<DomainTemplateOption[]>(cachedDomainOptions ?? DOMAIN_OPTIONS);

  useEffect(() => {
    let cancelled = false;
    fetchDomainTemplates().then((items) => {
      if (!cancelled) setTemplates(items);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return templates;
}

export function domainPrompt(vertical: DomainId | undefined): string {
  return domainTemplate(vertical).placeholderPrompt;
}

export function domainTemplate(vertical: DomainId | undefined): DomainTemplateOption {
  return DOMAIN_OPTIONS.find((item) => item.id === (vertical ?? 'generic')) ?? DOMAIN_OPTIONS[0];
}

export function domainTemplateFromList(templates: DomainTemplateOption[], vertical: DomainId | undefined): DomainTemplateOption {
  return templates.find((item) => item.id === (vertical ?? 'generic')) ?? templates[0] ?? DOMAIN_OPTIONS[0];
}

export function domainPromptFromList(templates: DomainTemplateOption[], vertical: DomainId | undefined): string {
  return domainTemplateFromList(templates, vertical).placeholderPrompt;
}

export function domainLabel(vertical: DomainId | undefined): string {
  return domainTemplate(vertical).label;
}

export function providerLabel(provider: CloudProviderId | undefined): string {
  return PROVIDERS.find((item) => item.id === (provider ?? 'hf-jobs'))?.label ?? PROVIDERS[0].label;
}
