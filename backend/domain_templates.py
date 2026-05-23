"""Built-in domain templates for Phase 8 workflow context.

The backend owns this registry so agent instructions are rendered from stable
domain ids instead of frontend labels or brittle route-level string checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_DOMAIN_ID = "generic"
DEFAULT_PROVIDER_ID = "hf-jobs"


@dataclass(frozen=True)
class StarterKit:
    starter_kit_id: str
    starter_prompts: tuple[str, ...]
    dataset_guidance: tuple[str, ...]
    recommended_datasets: tuple[str, ...]
    recommended_base_models: tuple[str, ...]
    evaluation_rubric: tuple[str, ...]
    metrics: tuple[str, ...]
    expected_columns: tuple[str, ...]
    expected_labels: tuple[str, ...]
    compliance_notes: tuple[str, ...]
    workflow_steps: tuple[str, ...]


@dataclass(frozen=True)
class DomainTemplate:
    domain_id: str
    label: str
    detail: str
    placeholder_prompt: str
    context_instructions: tuple[str, ...]
    suggested_datasets: tuple[str, ...] = ()
    evaluation_hints: tuple[str, ...] = ()
    compliance_notes: tuple[str, ...] = ()
    fast_path_instructions: tuple[str, ...] = ()
    starter_kit: StarterKit | None = None

    def to_api_dict(self) -> dict[str, Any]:
        return asdict(self)


_PHASE6_CALL_CENTER_DATASET = (
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)
BUILTIN_STARTER_KIT_IDS = ("itr", "gst", "fieldops", "call-center")


DOMAIN_TEMPLATES: dict[str, DomainTemplate] = {
    "generic": DomainTemplate(
        domain_id="generic",
        label="Generic",
        detail="Open-ended ML workflow",
        placeholder_prompt="Describe the dataset, target metric, and model goal.",
        context_instructions=(
            "Use the selected workflow context unless the user explicitly asks to change it.",
            "Clarify missing dataset, metric, label, and deployment constraints before proposing spend.",
        ),
        suggested_datasets=(),
        evaluation_hints=(
            "Choose task-appropriate metrics and explain baseline expectations.",
        ),
        compliance_notes=(
            "Do not expose secrets, tokens, private repo contents, or personal data.",
        ),
    ),
    "itr": DomainTemplate(
        domain_id="itr",
        label="ITR",
        detail="Tax forms and income signals",
        placeholder_prompt="Upload or describe ITR-style records, labels, and validation rules.",
        context_instructions=(
            "Treat the task as income-tax return document or structured tax-record modeling.",
            "Ask for schema, assessment year, target label, and validation rules when missing.",
        ),
        suggested_datasets=("Customer-provided private ITR dataset",),
        evaluation_hints=(
            "Report field-level extraction quality, validation-rule pass rates, and task-specific accuracy.",
        ),
        compliance_notes=(
            "Redact PAN, Aadhaar, bank account numbers, phone numbers, addresses, and tax identifiers.",
            "Keep customer-uploaded ITR datasets private unless the user explicitly requests otherwise.",
        ),
        starter_kit=StarterKit(
            starter_kit_id="itr",
            starter_prompts=(
                "Build an ITR assistant that extracts declared income, deductions, and filing status from private tax records.",
                "Fine-tune a classifier to flag ITR records needing manual review from structured validation signals.",
            ),
            dataset_guidance=(
                "Bring a private, permissioned ITR dataset; do not use public synthetic rows as a substitute for production quality.",
                "Prefer CSV, JSONL, or Parquet with one filing or document segment per row and clear label provenance.",
            ),
            recommended_datasets=("Customer-provided private ITR dataset",),
            recommended_base_models=(
                "Qwen/Qwen2.5-1.5B-Instruct",
                "microsoft/Phi-3.5-mini-instruct",
            ),
            evaluation_rubric=(
                "Field extraction must preserve numeric amounts and assessment-year references.",
                "Validation outcomes should be explainable against provided tax rules.",
                "Sample high-risk outputs for human review before using them operationally.",
            ),
            metrics=(
                "field-level F1",
                "exact match for identifiers after redaction",
                "validation-rule pass rate",
                "manual-review precision/recall",
            ),
            expected_columns=(
                "assessment_year",
                "income_fields",
                "deduction_fields",
                "filing_status",
                "target_label",
                "redaction_status",
            ),
            expected_labels=(
                "filing_status",
                "review_required",
                "field_extraction_target",
            ),
            compliance_notes=(
                "Redact PAN, Aadhaar, bank account numbers, phone numbers, addresses, and tax identifiers before training.",
                "Keep raw returns and derived labels in private repos unless the user explicitly approves publishing.",
            ),
            workflow_steps=(
                "Confirm schema, assessment year coverage, and target label.",
                "Run redaction checks before upload or training.",
                "Choose extraction, classification, or validation as the primary task.",
                "Propose a small HF Jobs run only after dataset and metric constraints are clear.",
            ),
        ),
    ),
    "gst": DomainTemplate(
        domain_id="gst",
        label="GST",
        detail="Invoice and compliance datasets",
        placeholder_prompt="Frame the GST prediction, anomaly, or extraction task.",
        context_instructions=(
            "Treat the task as GST invoice, filing, anomaly detection, or compliance extraction work.",
            "Ask for invoice schema, tax periods, labels, and jurisdiction-specific rules when missing.",
        ),
        suggested_datasets=("Customer-provided private GST invoice or return dataset",),
        evaluation_hints=(
            "Use extraction F1, anomaly precision/recall, reconciliation accuracy, or validation-rule pass rates.",
        ),
        compliance_notes=(
            "Redact GSTINs, invoice numbers, addresses, bank details, phone numbers, and emails.",
            "Do not publish customer GST datasets without explicit user approval.",
        ),
        starter_kit=StarterKit(
            starter_kit_id="gst",
            starter_prompts=(
                "Train a GST invoice model to classify reconciliation status and explain mismatch reasons.",
                "Build a GST extraction workflow for invoice fields, tax rates, and filing-period checks.",
            ),
            dataset_guidance=(
                "Use a private GST invoice, e-way bill, or return dataset with permissioned business records.",
                "Include filing period, invoice line items, tax components, and a clear anomaly or extraction label.",
            ),
            recommended_datasets=(
                "Customer-provided private GST invoice or return dataset",
            ),
            recommended_base_models=(
                "Qwen/Qwen2.5-1.5B-Instruct",
                "google/gemma-2-2b-it",
            ),
            evaluation_rubric=(
                "Extraction outputs should reconcile taxable value, CGST, SGST, IGST, and total invoice amount.",
                "Anomaly decisions should separate data-entry errors from compliance-risk patterns.",
                "Report business-cost examples for false positives and false negatives.",
            ),
            metrics=(
                "extraction F1",
                "anomaly precision/recall",
                "reconciliation accuracy",
                "validation-rule pass rate",
            ),
            expected_columns=(
                "filing_period",
                "invoice_text",
                "seller_gstin_hash",
                "buyer_gstin_hash",
                "taxable_value",
                "tax_components",
                "target_label",
            ),
            expected_labels=(
                "reconciled",
                "mismatch_reason",
                "compliance_risk",
            ),
            compliance_notes=(
                "Redact or hash GSTINs, invoice numbers, addresses, bank details, phone numbers, and emails.",
                "Keep customer invoices private and avoid publishing derived rows without explicit approval.",
            ),
            workflow_steps=(
                "Confirm invoice schema, filing periods, and jurisdiction rules.",
                "Decide whether extraction, anomaly detection, or reconciliation is primary.",
                "Validate redaction and private dataset location before training.",
                "Use a small baseline run before scaling to full historical filings.",
            ),
        ),
    ),
    "fieldops": DomainTemplate(
        domain_id="fieldops",
        label="FieldOps",
        detail="Inspections, tickets, and route work",
        placeholder_prompt="Summarize the field workflow, outcome label, and operational constraints.",
        context_instructions=(
            "Treat the task as field operations, inspections, tickets, routing, or workforce support.",
            "Ask for workflow stage, SLA, location granularity, and target outcome when missing.",
        ),
        suggested_datasets=(
            "Customer-provided field ticket, inspection, or route dataset",
        ),
        evaluation_hints=(
            "Track classification quality, SLA/routing accuracy, calibration, and operational error costs.",
        ),
        compliance_notes=(
            "Redact customer names, addresses, phone numbers, GPS coordinates, and employee identifiers.",
        ),
        starter_kit=StarterKit(
            starter_kit_id="fieldops",
            starter_prompts=(
                "Fine-tune a field operations classifier to predict ticket priority and next best action.",
                "Build a field inspection assistant that summarizes visit notes and flags SLA risk.",
            ),
            dataset_guidance=(
                "Use private ticket, inspection, route, or workforce datasets with timestamps and outcome labels.",
                "Include SLA, location granularity, technician notes, and final resolution where available.",
            ),
            recommended_datasets=(
                "Customer-provided field ticket, inspection, or route dataset",
            ),
            recommended_base_models=(
                "Qwen/Qwen2.5-1.5B-Instruct",
                "microsoft/Phi-3.5-mini-instruct",
            ),
            evaluation_rubric=(
                "Predictions should be calibrated enough for queueing or routing decisions.",
                "Summaries must preserve safety issues, blockers, and customer commitments.",
                "Measure impact separately for urgent SLA breaches and routine work.",
            ),
            metrics=(
                "priority classification F1",
                "SLA-risk precision/recall",
                "routing accuracy",
                "calibration error",
            ),
            expected_columns=(
                "ticket_id",
                "created_at",
                "location_bucket",
                "issue_description",
                "technician_notes",
                "sla_status",
                "target_label",
            ),
            expected_labels=(
                "priority",
                "next_best_action",
                "sla_risk",
            ),
            compliance_notes=(
                "Redact customer names, addresses, phone numbers, exact GPS coordinates, and employee identifiers.",
                "Bucket locations where exact coordinates are not necessary for the task.",
            ),
            workflow_steps=(
                "Confirm operational workflow stage and target decision.",
                "Check redaction for customer, employee, and location identifiers.",
                "Define the error cost for routing, SLA, and priority mistakes.",
                "Start with a bounded dataset slice before proposing larger jobs.",
            ),
        ),
    ),
    "call-center": DomainTemplate(
        domain_id="call-center",
        label="Call Center",
        detail="Conversations, QA, and intent data",
        placeholder_prompt="Share the call transcript objective, labels, and evaluation rubric.",
        context_instructions=(
            "Treat the task as support conversations, QA, intent routing, summarization, or agent-assist work.",
            "Ask for transcript format, labels, language mix, and quality rubric when missing.",
        ),
        suggested_datasets=(_PHASE6_CALL_CENTER_DATASET,),
        evaluation_hints=(
            "Use exact-match or F1 for labels, rubric pass rates for QA, and human-review samples for generation.",
        ),
        compliance_notes=(
            "Redact names, emails, phone numbers, account numbers, addresses, and payment details.",
        ),
        fast_path_instructions=(
            "Phase 6 approval-before-spend fast path:",
            "- Keep research bounded: inspect the selected/candidate dataset directly, then proceed to a minimal job proposal.",
            "- Do not call `research`, `explore_hf_docs`, `fetch_hf_docs`, GitHub search, or repo-read tools on this smoke path.",
            "- Prefer a small base model and HF Jobs with billable hardware such as `t4-small` so cost/manual approval guardrails are visible.",
            "- After dataset inspection, the next assistant turn must be an `hf_jobs` tool call; do not write a long pre-flight explanation first.",
            "- When the minimal job proposal is ready, call `hf_jobs` with compact JSON arguments to request approval and trigger the approval_required event; do not ask for approval only in chat.",
            '- Use the `hf_jobs` schema directly, for example `{"operation":"run","command":["python","-c","print(\\"approval smoke\\")"],"image":"python:3.12","hardware_flavor":"t4-small","timeout":"30m"}`.',
            "- Stop when approval is required; do not approve, launch, or continue into billable spend automatically.",
            "- Do not create, write to, or mutate Hub repos for this smoke path; avoid `hf_repo_git`, `hf_repo_create`, and similar repo-write tools.",
            "- Do not create a sandbox for this smoke path unless the user explicitly asks to test code first.",
            "- If sandbox work is required, call `sandbox_create` before `bash`, `read`, `write`, or `edit`.",
        ),
        starter_kit=StarterKit(
            starter_kit_id="call-center",
            starter_prompts=(
                "Fine-tune a customer support assistant for intent routing, answer quality, and safe escalation.",
                "Build a call-center QA model that scores transcripts against a human review rubric.",
            ),
            dataset_guidance=(
                "Use private transcripts or a support dataset with utterances, roles, intents, and QA labels.",
                "For a smoke path, the Bitext customer support dataset can be inspected before proposing a minimal HF Jobs run.",
            ),
            recommended_datasets=(_PHASE6_CALL_CENTER_DATASET,),
            recommended_base_models=(
                "Qwen/Qwen2.5-1.5B-Instruct",
                "google/gemma-2-2b-it",
            ),
            evaluation_rubric=(
                "Intent routing should prefer safe escalation over unsupported confident answers.",
                "Generated responses should be scored for correctness, empathy, policy adherence, and PII handling.",
                "QA outputs should include a reason tied to transcript evidence.",
            ),
            metrics=(
                "intent accuracy",
                "macro F1",
                "rubric pass rate",
                "human-review agreement",
            ),
            expected_columns=(
                "conversation_id",
                "turn_text",
                "speaker_role",
                "intent",
                "resolution_status",
                "qa_score",
                "target_response",
            ),
            expected_labels=(
                "intent",
                "escalation_required",
                "qa_score",
                "resolution_status",
            ),
            compliance_notes=(
                "Redact PII such as names, emails, phone numbers, account numbers, addresses, and payment details.",
                "Keep private transcripts private unless the user explicitly approves publication.",
            ),
            workflow_steps=(
                "Confirm transcript format, roles, language mix, and labels.",
                "Validate PII redaction before upload or training.",
                "Choose intent routing, QA scoring, summarization, or agent assist as the primary task.",
                "For HF Jobs, stop at approval_required before any billable launch.",
            ),
        ),
    ),
}


PROVIDER_LABELS: dict[str, str] = {
    "hf-jobs": "HF Jobs",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
}


def normalize_domain_id(value: Any) -> str:
    return (
        value
        if isinstance(value, str) and value in DOMAIN_TEMPLATES
        else DEFAULT_DOMAIN_ID
    )


def normalize_provider_id(value: Any) -> str:
    return (
        value
        if isinstance(value, str) and value in PROVIDER_LABELS
        else DEFAULT_PROVIDER_ID
    )


def normalize_starter_kit_id(value: Any, domain_id: Any = None) -> str | None:
    """Return the backend-owned starter kit for a built-in domain, if any."""
    normalized_domain = normalize_domain_id(domain_id)
    if normalized_domain in BUILTIN_STARTER_KIT_IDS:
        return normalized_domain
    if isinstance(value, str) and value in BUILTIN_STARTER_KIT_IDS:
        return value
    return None


def get_domain_template(domain_id: Any) -> DomainTemplate:
    return DOMAIN_TEMPLATES[normalize_domain_id(domain_id)]


def list_domain_templates() -> list[DomainTemplate]:
    return [
        DOMAIN_TEMPLATES[key]
        for key in ("generic", "itr", "gst", "fieldops", "call-center")
    ]


def _clean_dataset_repo(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    repo = value.strip()
    return repo or None


def _metadata_from_legacy_context(context: str) -> dict[str, str | None]:
    lower = context.lower()
    domain_id = DEFAULT_DOMAIN_ID
    for candidate, template in DOMAIN_TEMPLATES.items():
        if (
            f"vertical: {template.label.lower()}" in lower
            or f"domain: {template.label.lower()}" in lower
        ):
            domain_id = candidate
            break

    provider_id = DEFAULT_PROVIDER_ID
    for candidate, label in PROVIDER_LABELS.items():
        if f"compute provider: {label.lower()}" in lower:
            provider_id = candidate
            break

    dataset_repo = None
    for line in context.splitlines():
        if line.lower().strip().startswith("- dataset repo:"):
            dataset_repo = line.split(":", 1)[1].strip() or None
            break

    return {
        "domain_id": domain_id,
        "starter_kit_id": normalize_starter_kit_id(None, domain_id),
        "provider_id": provider_id,
        "dataset_repo": dataset_repo,
    }


def normalize_workflow_metadata(context: Any) -> dict[str, str | None]:
    if isinstance(context, str):
        return _metadata_from_legacy_context(context)
    if not isinstance(context, dict):
        return {
            "domain_id": DEFAULT_DOMAIN_ID,
            "starter_kit_id": None,
            "provider_id": DEFAULT_PROVIDER_ID,
            "dataset_repo": None,
        }

    domain_id = normalize_domain_id(context.get("domain_id") or context.get("vertical"))
    dataset = context.get("dataset_repo") or context.get("datasetRepo")
    if not dataset and isinstance(context.get("dataset"), dict):
        dataset = context["dataset"].get("repoId") or context["dataset"].get("repo_id")

    return {
        "domain_id": domain_id,
        "starter_kit_id": normalize_starter_kit_id(
            context.get("starter_kit_id") or context.get("starterKitId"), domain_id
        ),
        "provider_id": normalize_provider_id(
            context.get("provider_id") or context.get("provider")
        ),
        "dataset_repo": _clean_dataset_repo(dataset),
    }


def render_workflow_context(context: Any) -> str:
    if isinstance(context, str) and context.strip():
        metadata = normalize_workflow_metadata(context)
    elif isinstance(context, dict):
        metadata = normalize_workflow_metadata(context)
    else:
        return ""

    template = get_domain_template(metadata["domain_id"])
    provider_id = normalize_provider_id(metadata["provider_id"])
    dataset_repo = metadata.get("dataset_repo")

    lines = [
        "Selected ML workflow context:",
        f"- Domain: {template.label}",
        f"- Compute provider: {PROVIDER_LABELS[provider_id]}",
    ]
    if dataset_repo:
        lines.append(f"- Dataset repo: {dataset_repo}")

    lines.extend(f"- {item}" for item in template.context_instructions)
    if template.suggested_datasets:
        lines.append("Suggested datasets:")
        lines.extend(f"- {item}" for item in template.suggested_datasets)
    if template.evaluation_hints:
        lines.append("Evaluation hints:")
        lines.extend(f"- {item}" for item in template.evaluation_hints)
    if template.compliance_notes:
        lines.append("Compliance/redaction:")
        lines.extend(f"- {item}" for item in template.compliance_notes)
    if provider_id != "hf-jobs":
        lines.extend(_render_plan_only_provider(provider_id))
    if template.fast_path_instructions and provider_id == "hf-jobs":
        lines.extend(_render_fast_path(template, dataset_repo))
    if template.starter_kit:
        lines.extend(_render_starter_kit(template.starter_kit))

    return "\n".join(lines)


def _render_starter_kit(starter_kit: StarterKit) -> list[str]:
    lines = [f"Starter kit: {starter_kit.starter_kit_id}"]
    sections = (
        ("Starter prompts", starter_kit.starter_prompts),
        ("Dataset guidance", starter_kit.dataset_guidance),
        ("Recommended datasets", starter_kit.recommended_datasets),
        ("Recommended base models", starter_kit.recommended_base_models),
        ("Evaluation rubric", starter_kit.evaluation_rubric),
        ("Metrics", starter_kit.metrics),
        ("Expected dataset columns", starter_kit.expected_columns),
        ("Expected labels", starter_kit.expected_labels),
        ("Starter-kit compliance", starter_kit.compliance_notes),
        ("Workflow steps", starter_kit.workflow_steps),
    )
    for title, items in sections:
        if items:
            lines.append(f"{title}:")
            lines.extend(f"- {item}" for item in items)
    return lines


def _render_plan_only_provider(provider_id: str) -> list[str]:
    label = PROVIDER_LABELS[provider_id]
    return [
        f"{label} plan-only preview:",
        f"- Treat {label} as an active plan-only provider for Phase 10.",
        f"- Do not create {label} resources, submit real cloud jobs, or imply cloud spend.",
        "- Use the `hf_jobs` tool with `operation: plan` or `operation: validate` for "
        "provider-specific training plans and credential readiness checks.",
        "- If the tool call uses `operation: run`, it must remain a dry-run plan with no "
        "billable provider action.",
        "- Real submit operations remain disabled until explicit credentials, region/project "
        "configuration, and manual approval policy are added.",
    ]


def _render_fast_path(template: DomainTemplate, dataset_repo: str | None) -> list[str]:
    lines = list(template.fast_path_instructions)
    dataset_instruction = (
        "- Use the selected dataset repo exactly; do not silently substitute it."
        if dataset_repo
        else (
            "- If no dataset repo is selected and the user asks for a Hugging Face customer support dataset, "
            f"use `{_PHASE6_CALL_CENTER_DATASET}` as the candidate dataset and link it explicitly."
        )
    )
    return [lines[0], dataset_instruction, *lines[1:]]
