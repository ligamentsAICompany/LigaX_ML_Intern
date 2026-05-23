"""Deterministic one-run auto fine-tuning pipeline for HF Jobs."""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from agent.config import Config
from agent.core.cost_estimation import estimate_tool_cost
from agent.core.redact import scrub, scrub_string
from agent.core.script_smoke import run_script_smoke
from agent.core.session import Event
from agent.core.training_templates import render_sft_training_script

AUTO_FINETUNE_COST_CAP_USD = 5.0
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_HARDWARE_FLAVOR = "t4-small"
DEFAULT_TIMEOUT = "3h"
DEFAULT_MAX_LENGTH = 1024

_FINE_TUNE_RE = re.compile(r"\b(?:fine[-\s]?tune|finetune|sft)\b", re.I)
_DATASET_URL_RE = re.compile(
    r"https?://huggingface\.co/datasets/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
)
_REPO_RE = re.compile(r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\b")
_JOB_URL_RE = re.compile(r"https://huggingface\.co/jobs/[^\s)]+")
_MODEL_URL_RE = re.compile(r"AUTO_FINETUNE_MODEL_URL=(https://huggingface\.co/[^\s]+)")


@dataclass(frozen=True)
class AutoFineTuneConfig:
    provider_id: str = "hf-jobs"
    cost_cap_usd: float = AUTO_FINETUNE_COST_CAP_USD
    max_retries: int = 1
    base_model: str = DEFAULT_BASE_MODEL
    hardware_flavor: str = DEFAULT_HARDWARE_FLAVOR
    timeout: str = DEFAULT_TIMEOUT
    max_length: int = DEFAULT_MAX_LENGTH


@dataclass(frozen=True)
class AutoFineTunePlan:
    dataset_repo: str
    output_model_repo: str
    model_repo_url: str
    provider_id: str
    job_args: dict[str, Any]
    estimated_job_cost_usd: float
    estimated_total_cost_usd: float
    cost_cap_usd: float
    max_retries: int
    approval_required: bool = False


@dataclass(frozen=True)
class AutoFineTuneBlock:
    error_code: str
    message: str
    approval_required: bool = False


@dataclass(frozen=True)
class AutoFineTuneJobResult:
    ok: bool
    output: str
    job_url: str | None = None
    final_status: str | None = None


@dataclass(frozen=True)
class AutoFineTuneRunResult:
    success: bool
    message: str
    approval_required: bool
    model_repo_url: str | None = None
    job_url: str | None = None
    eval_result: str | None = None
    error_code: str | None = None


Executor = Callable[[dict[str, Any]], Awaitable[AutoFineTuneJobResult | Any]]


def resolve_auto_finetune_config(config: Config | None) -> AutoFineTuneConfig:
    raw_cap = getattr(config, "auto_finetune_cost_cap_usd", AUTO_FINETUNE_COST_CAP_USD)
    raw_retries = getattr(config, "auto_finetune_max_retries", 1)
    try:
        cap = min(float(raw_cap), AUTO_FINETUNE_COST_CAP_USD)
    except (TypeError, ValueError):
        cap = AUTO_FINETUNE_COST_CAP_USD
    try:
        max_retries = max(0, min(int(raw_retries), 2))
    except (TypeError, ValueError):
        max_retries = 1
    return AutoFineTuneConfig(cost_cap_usd=cap, max_retries=max_retries)


def credential_readiness(env: Mapping[str, str] | None = None) -> dict[str, bool]:
    env = env or os.environ
    return {
        "hf_token_configured": bool(env.get("HF_TOKEN")),
        "openai_api_key_configured": bool(env.get("OPENAI_API_KEY")),
    }


def is_auto_finetune_intent(text: str | None) -> bool:
    return bool(text and _FINE_TUNE_RE.search(text))


def resolve_dataset_repo(
    text: str | None, context: Mapping[str, Any] | None
) -> str | None:
    context = context or {}
    dataset_repo = context.get("dataset_repo")
    if isinstance(dataset_repo, str) and dataset_repo.strip():
        return _normalize_repo_id(dataset_repo)

    text = text or ""
    url_match = _DATASET_URL_RE.search(text)
    if url_match:
        return _normalize_repo_id(url_match.group(1))

    for match in _REPO_RE.finditer(text):
        candidate = match.group(1)
        if "huggingface.co" not in candidate:
            return _normalize_repo_id(candidate)
    return None


def build_auto_finetune_job_args(
    *,
    dataset_repo: str,
    output_model_repo: str,
    config: AutoFineTuneConfig,
) -> dict[str, Any]:
    script = render_sft_training_script(
        dataset_repo=dataset_repo,
        output_model_repo=output_model_repo,
        base_model=config.base_model,
        max_length=config.max_length,
        trackio_project="",
        trackio_run_name=_slug(output_model_repo.split("/", 1)[-1]),
    )
    dependencies = [
        "accelerate",
        "datasets",
        "huggingface-hub",
        "peft",
        "torch",
        "transformers",
        "trl",
    ]
    args: dict[str, Any] = {
        "operation": "run",
        "provider_id": config.provider_id,
        "script": script,
        "dependencies": dependencies,
        "hardware_flavor": config.hardware_flavor,
        "timeout": config.timeout,
        "secrets": {"HF_TOKEN": "$HF_TOKEN"},
        "env": {"HF_HUB_DISABLE_PROGRESS_BARS": "1"},
        "provenance": {
            "auto_finetune": True,
            "provider": config.provider_id,
            "base_model": config.base_model,
            "dataset_repo": dataset_repo,
            "output_model_repo": output_model_repo,
            "hardware": config.hardware_flavor,
            "timeout": config.timeout,
            "cost_cap_usd": config.cost_cap_usd,
        },
    }
    smoke = run_script_smoke(script, job_args=args)
    args["script_smoke"] = smoke.to_dict()
    return args


class AutoFineTunePipeline:
    """One-run state machine for template-based HF Jobs fine-tuning."""

    def __init__(
        self,
        *,
        config: Config | None = None,
        executor: Executor | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.config = resolve_auto_finetune_config(config)
        self.executor = executor
        self.env = env or os.environ

    async def prepare(
        self, text: str, context: Mapping[str, Any] | None
    ) -> AutoFineTunePlan | AutoFineTuneBlock:
        dataset_repo = resolve_dataset_repo(text, context)
        if not dataset_repo:
            return AutoFineTuneBlock(
                error_code="auto_finetune_dataset_missing",
                message=(
                    "I can auto fine-tune only when the dataset is attached or "
                    "mentioned as a Hugging Face dataset repo."
                ),
            )

        strategy_block = _strategy_block_message(context)
        if strategy_block:
            return AutoFineTuneBlock(
                error_code="auto_finetune_strategy_blocked",
                message=strategy_block,
                approval_required=True,
            )

        namespace = self._namespace()
        output_model_repo = (
            f"{namespace}/{_slug(dataset_repo.split('/', 1)[-1])}-auto-sft"
        )
        job_args = build_auto_finetune_job_args(
            dataset_repo=dataset_repo,
            output_model_repo=output_model_repo,
            config=self.config,
        )
        smoke = job_args["script_smoke"]
        if not smoke.get("passed"):
            return AutoFineTuneBlock(
                error_code="auto_finetune_smoke_failed",
                message="Auto fine-tune blocked by mandatory script smoke validation.",
            )

        estimate = await estimate_tool_cost("hf_jobs", job_args)
        if estimate.estimated_cost_usd is None:
            return AutoFineTuneBlock(
                error_code="auto_finetune_cost_unknown",
                message=estimate.block_reason
                or "Could not estimate HF Jobs cost safely.",
            )

        total_cost = round(
            float(estimate.estimated_cost_usd) * (self.config.max_retries + 1), 4
        )
        if total_cost > self.config.cost_cap_usd:
            return AutoFineTuneBlock(
                error_code="auto_finetune_cost_cap_exceeded",
                message=(
                    f"Auto fine-tune blocked: estimated total spend ${total_cost:.2f} "
                    f"including retries exceeds the ${self.config.cost_cap_usd:.2f} hard cap."
                ),
            )

        return AutoFineTunePlan(
            dataset_repo=dataset_repo,
            output_model_repo=output_model_repo,
            model_repo_url=f"https://huggingface.co/{output_model_repo}",
            provider_id=self.config.provider_id,
            job_args=job_args,
            estimated_job_cost_usd=float(estimate.estimated_cost_usd),
            estimated_total_cost_usd=total_cost,
            cost_cap_usd=self.config.cost_cap_usd,
            max_retries=self.config.max_retries,
        )

    async def run(
        self,
        session: Any,
        text: str,
        context: Mapping[str, Any] | None = None,
    ) -> AutoFineTuneRunResult:
        await self._progress(session, "resolving_dataset", "Resolving dataset")
        plan_or_block = await self.prepare(text, context)
        if isinstance(plan_or_block, AutoFineTuneBlock):
            await self._progress(
                session,
                "blocked",
                plan_or_block.message,
                error_code=plan_or_block.error_code,
            )
            await session.send_event(
                Event(
                    event_type="assistant_message",
                    data={"content": plan_or_block.message},
                )
            )
            return AutoFineTuneRunResult(
                success=False,
                message=plan_or_block.message,
                approval_required=plan_or_block.approval_required,
                error_code=plan_or_block.error_code,
            )

        plan = plan_or_block
        await self._emit_plan(session, plan)
        await self._progress(
            session,
            "preflight_passed",
            "Preflight and cost cap passed",
            estimated_total_cost_usd=plan.estimated_total_cost_usd,
        )

        attempt = 0
        current_args = dict(plan.job_args)
        last_output = ""
        while attempt <= plan.max_retries:
            tool_call_id = f"auto-finetune:{uuid.uuid4().hex[:12]}"
            await session.send_event(
                Event(
                    event_type="tool_call",
                    data={
                        "tool": "hf_jobs",
                        "tool_call_id": tool_call_id,
                        "arguments": scrub(
                            {**current_args, "script": "[template script]"}
                        ),
                    },
                )
            )
            await self._progress(
                session,
                "submitting_job" if attempt == 0 else "retrying_job",
                f"Submitting HF Jobs training attempt {attempt + 1}",
                attempt=attempt + 1,
                max_attempts=plan.max_retries + 1,
            )

            result = await self._execute(session, current_args)
            last_output = scrub_string(str(result.output))
            await session.send_event(
                Event(
                    event_type="tool_output",
                    data={
                        "tool": "hf_jobs",
                        "tool_call_id": tool_call_id,
                        "output": last_output,
                        "success": bool(result.ok),
                    },
                )
            )

            if result.ok:
                model_url = _extract_model_url(last_output) or plan.model_repo_url
                job_url = result.job_url or _extract_job_url(last_output)
                eval_result = _extract_eval(last_output)
                message = _final_message(model_url, job_url, eval_result)
                await self._progress(
                    session,
                    "completed",
                    "Fine-tuned model is ready",
                    model_repo_url=model_url,
                    job_url=job_url,
                    eval_result=eval_result,
                )
                await session.send_event(
                    Event(event_type="assistant_message", data={"content": message})
                )
                return AutoFineTuneRunResult(
                    success=True,
                    message=message,
                    approval_required=False,
                    model_repo_url=model_url,
                    job_url=job_url,
                    eval_result=eval_result,
                )

            repaired_args = apply_known_training_repair(current_args, last_output)
            if repaired_args is None or attempt >= plan.max_retries:
                break
            smoke = run_script_smoke(repaired_args["script"], job_args=repaired_args)
            if not smoke.passed:
                break
            repaired_args["script_smoke"] = smoke.to_dict()
            current_args = repaired_args
            attempt += 1

        message = "Auto fine-tune failed after bounded retries. " + last_output[:500]
        await self._progress(
            session, "failed", message, error_code="auto_finetune_failed"
        )
        await session.send_event(
            Event(event_type="assistant_message", data={"content": message})
        )
        return AutoFineTuneRunResult(
            success=False,
            message=message,
            approval_required=False,
            error_code="auto_finetune_failed",
            job_url=_extract_job_url(last_output),
        )

    async def _execute(
        self, session: Any, args: dict[str, Any]
    ) -> AutoFineTuneJobResult:
        if self.executor is not None:
            raw = await self.executor(args)
            if isinstance(raw, AutoFineTuneJobResult):
                return raw
            return AutoFineTuneJobResult(
                ok=bool(getattr(raw, "ok", False)),
                output=str(getattr(raw, "output", "")),
                job_url=getattr(raw, "job_url", None),
                final_status=getattr(raw, "final_status", None),
            )

        if getattr(session, "tool_router", None) is not None:
            output, ok = await session.tool_router.call_tool(
                "hf_jobs",
                args,
                session=session,
                tool_call_id=f"auto-finetune:{uuid.uuid4().hex[:12]}",
            )
            return AutoFineTuneJobResult(
                ok=ok,
                output=output,
                job_url=_extract_job_url(output),
            )

        raise RuntimeError("Auto fine-tune executor is unavailable.")

    def _namespace(self) -> str:
        namespace = self.env.get("HF_NAMESPACE")
        if namespace:
            return _slug(namespace)
        return "ligaments-dev"

    async def _emit_plan(self, session: Any, plan: AutoFineTunePlan) -> None:
        await session.send_event(
            Event(
                event_type="plan_update",
                data={
                    "plan": [
                        {
                            "id": "resolve",
                            "content": "Resolve and profile dataset",
                            "status": "completed",
                        },
                        {
                            "id": "template",
                            "content": "Render stable SFT template",
                            "status": "completed",
                        },
                        {
                            "id": "cap",
                            "content": "Enforce $5 HF Jobs cap",
                            "status": "completed",
                        },
                        {
                            "id": "submit",
                            "content": "Submit and monitor HF Job",
                            "status": "in_progress",
                        },
                        {
                            "id": "result",
                            "content": "Show model link and eval",
                            "status": "pending",
                        },
                    ]
                },
            )
        )

    async def _progress(
        self, session: Any, state: str, message: str, **data: Any
    ) -> None:
        await session.send_event(
            Event(
                event_type="auto_finetune_progress",
                data=scrub(
                    {
                        "state": state,
                        "message": message,
                        "provider_id": self.config.provider_id,
                        "approval_required": False,
                        "credential_readiness": credential_readiness(self.env),
                        **data,
                    }
                ),
            )
        )


def repair_known_training_error(script: str, error_text: str) -> str | None:
    """Apply deterministic repairs for known compatibility failures."""

    classification = classify_known_training_error(error_text)
    lowered = error_text.lower()
    repaired = script
    if classification == "trackio_sync_failure":
        repaired = _disable_trackio_reporting(repaired)
    if "max_seq_length" in lowered:
        repaired = repaired.replace("max_seq_length=", "max_length=")
    if "evaluation_strategy" in lowered:
        repaired = repaired.replace("evaluation_strategy=", "eval_strategy=")
    if "sfttrainer" in lowered and "tokenizer" in lowered:
        repaired = repaired.replace("tokenizer=tokenizer", "processing_class=tokenizer")
    if "project_name" in lowered or "run_name" in lowered:
        repaired = repaired.replace("project_name=", "project=").replace(
            "run_name=", "name="
        )
    return repaired if repaired != script else None


def classify_known_training_error(error_text: str) -> str | None:
    """Classify known training failures for deterministic bounded repair."""

    lowered = (error_text or "").lower()
    if (
        "rank_pattern" in lowered
        and "cannot write struct type" in lowered
        and (
            "trackio" in lowered
            or "integration_utils.py" in lowered
            or "on_push_begin" in lowered
        )
    ):
        return "trackio_sync_failure"
    return None


def apply_known_training_repair(
    job_args: Mapping[str, Any],
    error_text: str,
) -> dict[str, Any] | None:
    """Repair both inline script and metadata for known retry-safe failures."""

    script = str(job_args.get("script") or "")
    repaired_script = repair_known_training_error(script, error_text)
    if repaired_script is None:
        return None

    repaired_args = dict(job_args)
    repaired_args["script"] = repaired_script
    if classify_known_training_error(error_text) == "trackio_sync_failure":
        dependencies = repaired_args.get("dependencies")
        if isinstance(dependencies, list):
            repaired_args["dependencies"] = [
                dependency
                for dependency in dependencies
                if str(dependency).strip().lower() != "trackio"
            ]
        repaired_args.pop("trackio", None)
    return repaired_args


def _disable_trackio_reporting(script: str) -> str:
    repaired = script
    repaired = repaired.replace('"trackio", ', "").replace(', "trackio"', "")
    repaired = re.sub(r"(?m)^\s*import trackio\s*\n", "", repaired)
    repaired = re.sub(r"(?m)^\s*from trackio\b.*\n", "", repaired)
    repaired = re.sub(
        r"(?m)^\s*trackio\.(?:init|init_from_env)\(.*\)\s*\n?", "", repaired
    )
    repaired = re.sub(
        r"report_to\s*=\s*(?:\[\s*['\"]trackio['\"]\s*\]|['\"]trackio['\"])",
        "report_to=[]",
        repaired,
    )
    return repaired


def _strategy_block_message(context: Mapping[str, Any] | None) -> str | None:
    context = context or {}
    raw_profile = context.get("dataset_profile") or context.get("profile")
    if not isinstance(raw_profile, Mapping):
        return None
    strategy = raw_profile.get("strategy")
    if not isinstance(strategy, Mapping):
        return None
    strategy_name = str(strategy.get("strategy") or "").strip().lower()
    can_train = strategy.get("can_train_without_override")
    requires_override = strategy.get("requires_user_override_for_training")
    if strategy_name in {"rag", "hybrid", "data_needed"} or (
        can_train is False and requires_override is True
    ):
        override_message = str(strategy.get("override_message") or "").strip()
        if override_message:
            return override_message
        if strategy_name == "rag":
            return (
                "Auto fine-tune blocked: this dataset strategy recommends retrieval "
                "or reference lookup before any direct SFT run."
            )
        return (
            "Auto fine-tune blocked: the dataset strategy does not allow direct "
            "training without an explicit override."
        )
    return None


def _normalize_repo_id(value: str) -> str:
    return value.strip().removeprefix("datasets/").strip("/")


def _slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-._")
    return value or "auto-finetune"


def _extract_job_url(output: str) -> str | None:
    match = _JOB_URL_RE.search(output or "")
    return match.group(0) if match else None


def _extract_model_url(output: str) -> str | None:
    match = _MODEL_URL_RE.search(output or "")
    return match.group(1) if match else None


def _extract_eval(output: str) -> str | None:
    marker = "AUTO_FINETUNE_EVAL="
    for line in (output or "").splitlines():
        if line.startswith(marker):
            return line[len(marker) :].strip()
    return None


def _final_message(model_url: str, job_url: str | None, eval_result: str | None) -> str:
    lines = [
        "Auto fine-tune completed.",
        "",
        f"Fine-tuned model: {model_url}",
    ]
    if job_url:
        lines.append(f"HF Job: {job_url}")
    if eval_result:
        lines.append(f"Eval result: `{eval_result}`")
    return "\n".join(lines)
