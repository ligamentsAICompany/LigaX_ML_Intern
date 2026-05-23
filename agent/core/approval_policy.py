"""Central approval policy for billable and destructive tool operations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from agent.config import Config
from agent.core.cost_estimation import CostEstimate, estimate_tool_cost
from agent.core.script_smoke import (
    ScriptSmokeResult,
    coerce_script_smoke_result,
    format_script_smoke_result,
    is_inline_python_script,
    run_script_smoke,
    unverified_script_smoke_result,
)
from agent.core.trainability import assess_trainability
from agent.tools.jobs_tool import CPU_FLAVORS


@dataclass(frozen=True)
class ApprovalDecision:
    """Approval outcome for a parsed tool call."""

    approved: bool
    requires_approval: bool
    blocked: bool = False
    unknown_cost: bool = False
    auto_approved: bool = False
    auto_approval_blocked: bool = False
    block_reason: str | None = None
    estimated_cost_usd: float | None = None
    remaining_cap_usd: float | None = None
    billable: bool = False


def normalize_tool_operation(operation: Any) -> str:
    return str(operation or "").strip().lower()


def is_scheduled_operation(operation: Any) -> bool:
    return normalize_tool_operation(operation).startswith("scheduled ")


def _validate_tool_args(tool_args: dict) -> bool:
    args = tool_args.get("args", {})
    return isinstance(args, dict | type(None))


def _hf_hardware_flavor(tool_args: dict) -> str:
    return str(
        tool_args.get("hardware_flavor")
        or tool_args.get("flavor")
        or tool_args.get("hardware")
        or "cpu-basic"
    )


def _base_requires_approval(
    tool_name: str, tool_args: dict, config: Config | None = None
) -> bool:
    """Return legacy approval requirements before auto-approval policy."""
    if not _validate_tool_args(tool_args):
        return False

    if tool_name == "sandbox_create":
        return True

    if tool_name == "hf_jobs":
        operation = normalize_tool_operation(tool_args.get("operation"))
        if operation == "submit":
            return True
        if is_scheduled_operation(operation):
            return True
        if operation not in {"run", "uv"}:
            return False

        if _hf_hardware_flavor(tool_args) in CPU_FLAVORS:
            return bool(config.confirm_cpu_jobs) if config else True
        return True

    if tool_name == "hf_private_repos":
        operation = tool_args.get("operation", "")
        if operation == "upload_file":
            return not bool(config and config.auto_file_upload)
        if operation in {"create_repo"}:
            return True

    if tool_name == "hf_repo_files":
        return tool_args.get("operation", "") in {"upload", "delete"}

    if tool_name == "hf_repo_git":
        return tool_args.get("operation", "") in {
            "delete_branch",
            "delete_tag",
            "merge_pr",
            "create_repo",
            "update_repo",
        }

    return False


def legacy_needs_approval(
    tool_name: str, tool_args: dict, config: Config | None = None
) -> bool:
    """Synchronous compatibility predicate for existing callers/tests."""
    if tool_name == "hf_jobs" and _script_smoke_block_reason(tool_args):
        return True
    if tool_name == "hf_jobs" and _trainability_block_reason(tool_args):
        return True
    if tool_name == "hf_jobs" and _trainability_manual_reason(tool_args):
        return True
    if tool_name == "hf_jobs" and is_scheduled_operation(tool_args.get("operation")):
        return True
    if config and config.yolo_mode:
        return False
    return _base_requires_approval(tool_name, tool_args, config)


def _is_budgeted_cost_target(tool_name: str, tool_args: dict) -> bool:
    if tool_name == "sandbox_create":
        return True
    if tool_name != "hf_jobs":
        return False
    return normalize_tool_operation(tool_args.get("operation")) in {
        "run",
        "uv",
        "scheduled run",
        "scheduled uv",
        "submit",
    }


def _is_hf_training_operation(tool_args: dict) -> bool:
    operation = normalize_tool_operation(tool_args.get("operation"))
    if operation not in {"run", "uv", "scheduled run", "scheduled uv", "submit"}:
        return False

    text_parts: list[str] = []
    for key in ("script", "command", "image"):
        value = tool_args.get(key)
        if isinstance(value, list):
            text_parts.extend(str(item) for item in value)
        elif value is not None:
            text_parts.append(str(value))
    dependencies = tool_args.get("dependencies")
    if isinstance(dependencies, list):
        text_parts.extend(str(item) for item in dependencies)

    text = " ".join(text_parts).lower()
    if any(
        marker in text
        for marker in (
            "accelerate launch",
            "sfttrainer",
            "trainer.train",
            "trainingarguments",
            "unsloth",
        )
    ):
        return True
    if re.search(r"\b(?:trl|sft|dpo|grpo)\b", text):
        return True
    return bool(
        re.search(
            r"(^|[\\/\s_.-])(?:train|trainer|finetune|fine[-_]?tune)(?:\.py|[\\/\s_.-]|$)",
            text,
        )
    )


def _trainability_result_from_args(tool_args: dict) -> Mapping[str, Any] | None:
    for key in ("dataset_profile", "dataset", "dataset_metadata"):
        profile = tool_args.get(key)
        if not isinstance(profile, Mapping):
            continue
        if not _has_trainability_metadata(profile):
            return None
        return assess_trainability(profile).to_dict()

    for key in ("trainability", "trainability_gate", "trainability_result"):
        explicit = tool_args.get(key)
        if isinstance(explicit, Mapping) and _is_blocking_trainability_result(explicit):
            return explicit

    return None


def _has_trainability_metadata(profile: Mapping[str, Any]) -> bool:
    return any(
        key in profile
        for key in (
            "row_count",
            "columns",
            "format",
            "file_type",
            "sample_rows",
            "missing_fraction",
            "duplicate_fraction",
        )
    )


def _is_blocking_trainability_result(result: Mapping[str, Any]) -> bool:
    risk_level = str(result.get("risk_level") or "").lower()
    recommendation = str(result.get("recommendation") or "").lower()
    return risk_level == "high" and recommendation != "fine_tune"


def _trainability_block_reason(tool_args: dict) -> str | None:
    if not _is_hf_training_operation(tool_args):
        return None

    result = _trainability_result_from_args(tool_args)
    if not result:
        return None

    recommendation = str(result.get("recommendation") or "").lower()
    if not _is_blocking_trainability_result(result):
        return None

    reasons = result.get("reasons")
    reason_text = ""
    if isinstance(reasons, list) and reasons:
        reason_text = f" Reasons: {'; '.join(str(reason) for reason in reasons[:3])}"
    return (
        "Trainability Gate blocked this direct fine-tuning job because the "
        f"dataset is high risk and the recommended path is {recommendation or 'not fine-tune'}."
        f"{reason_text}"
    )


def _trainability_manual_reason(tool_args: dict) -> str | None:
    if not _is_hf_training_operation(tool_args):
        return None
    if _trainability_result_from_args(tool_args):
        return None
    return (
        "Trainability Gate requires manual approval because this looks like a "
        "training job but no authoritative dataset profile was supplied."
    )


def _script_smoke_result_from_args(tool_args: dict) -> ScriptSmokeResult | None:
    script = tool_args.get("script")
    if isinstance(script, str):
        if is_inline_python_script(script):
            return run_script_smoke(script, job_args=tool_args)

        explicit_failure = _failed_explicit_script_smoke(tool_args)
        if explicit_failure is not None:
            return explicit_failure

        return unverified_script_smoke_result(script, job_args=tool_args)

    for key in ("script_smoke", "preflight"):
        explicit = tool_args.get(key)
        if isinstance(explicit, Mapping):
            return coerce_script_smoke_result(explicit)

    return None


def _failed_explicit_script_smoke(tool_args: dict) -> ScriptSmokeResult | None:
    for key in ("script_smoke", "preflight"):
        explicit = tool_args.get(key)
        if not isinstance(explicit, Mapping):
            continue
        result = coerce_script_smoke_result(explicit)
        if not result.passed:
            return result
    return None


def _script_smoke_block_reason(tool_args: dict) -> str | None:
    operation = normalize_tool_operation(tool_args.get("operation"))
    if operation not in {"run", "uv", "scheduled run", "scheduled uv", "submit"}:
        return None

    result = _script_smoke_result_from_args(tool_args)
    if result is None or result.passed:
        return None
    return "Script smoke failed before HF Jobs spend.\n" + format_script_smoke_result(
        result
    )


def _cost_cap_usd(session: Any = None, config: Config | None = None) -> float | None:
    value = getattr(session, "auto_approval_cost_cap_usd", None)
    if value is None and config is not None:
        value = getattr(config, "auto_approval_cost_cap_usd", None)
    if value is None:
        return 0.0
    try:
        cap = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, cap)


def _estimated_spend_usd(session: Any = None) -> float:
    try:
        return float(getattr(session, "auto_approval_estimated_spend_usd", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def remaining_cost_cap_usd(
    session: Any = None,
    config: Config | None = None,
    *,
    reserved_spend_usd: float = 0.0,
) -> float | None:
    cap = _cost_cap_usd(session, config)
    if cap is None:
        return None
    return round(max(0.0, cap - _estimated_spend_usd(session) - reserved_spend_usd), 4)


def record_estimated_spend(session: Any, decision: ApprovalDecision) -> None:
    if not decision.billable or decision.estimated_cost_usd is None:
        return
    amount = float(decision.estimated_cost_usd)
    if hasattr(session, "add_auto_approval_estimated_spend"):
        session.add_auto_approval_estimated_spend(amount)
        return
    current = _estimated_spend_usd(session)
    session.auto_approval_estimated_spend_usd = round(current + amount, 4)


def _manual_for_estimate(
    estimate: CostEstimate,
    *,
    remaining_cap_usd: float | None,
) -> ApprovalDecision | None:
    if estimate.estimated_cost_usd is None:
        return ApprovalDecision(
            approved=False,
            requires_approval=True,
            unknown_cost=True,
            auto_approval_blocked=True,
            block_reason=estimate.block_reason or "Could not estimate the cost safely.",
            estimated_cost_usd=None,
            remaining_cap_usd=remaining_cap_usd,
            billable=estimate.billable,
        )
    if (
        remaining_cap_usd is not None
        and estimate.estimated_cost_usd > remaining_cap_usd
    ):
        return ApprovalDecision(
            approved=False,
            requires_approval=True,
            auto_approval_blocked=True,
            block_reason=(
                f"Estimated cost ${estimate.estimated_cost_usd:.2f} exceeds remaining "
                f"auto-approval cap ${remaining_cap_usd:.2f}."
            ),
            estimated_cost_usd=estimate.estimated_cost_usd,
            remaining_cap_usd=remaining_cap_usd,
            billable=estimate.billable,
        )
    return None


async def decide_tool_approval(
    tool_name: str,
    tool_args: dict,
    *,
    session: Any = None,
    config: Config | None = None,
    reserved_spend_usd: float = 0.0,
) -> ApprovalDecision:
    """Decide whether a tool call can run, needs approval, or is blocked."""
    effective_config = config or getattr(session, "config", None)
    script_smoke_block_reason = (
        _script_smoke_block_reason(tool_args) if tool_name == "hf_jobs" else None
    )
    trainability_block_reason = (
        _trainability_block_reason(tool_args) if tool_name == "hf_jobs" else None
    )
    trainability_manual_reason = (
        _trainability_manual_reason(tool_args) if tool_name == "hf_jobs" else None
    )
    if trainability_block_reason:
        return ApprovalDecision(
            approved=False,
            requires_approval=True,
            blocked=True,
            auto_approval_blocked=True,
            block_reason=trainability_block_reason,
        )
    if trainability_manual_reason:
        return ApprovalDecision(
            approved=False,
            requires_approval=True,
            auto_approval_blocked=True,
            block_reason=trainability_manual_reason,
            billable=True,
        )
    if script_smoke_block_reason:
        return ApprovalDecision(
            approved=False,
            requires_approval=True,
            blocked=True,
            auto_approval_blocked=True,
            block_reason=script_smoke_block_reason,
            billable=True,
        )

    base_requires = _base_requires_approval(tool_name, tool_args, effective_config)
    yolo_enabled = bool(effective_config and effective_config.yolo_mode)

    if tool_name == "hf_jobs" and is_scheduled_operation(tool_args.get("operation")):
        return ApprovalDecision(
            approved=False,
            requires_approval=True,
            auto_approval_blocked=yolo_enabled,
            block_reason="Scheduled HF jobs always require manual approval.",
        )

    if not base_requires:
        return ApprovalDecision(approved=True, requires_approval=False)

    if not yolo_enabled:
        return ApprovalDecision(approved=False, requires_approval=True)

    if _is_budgeted_cost_target(tool_name, tool_args):
        estimate = await estimate_tool_cost(tool_name, tool_args, session=session)
        remaining = remaining_cost_cap_usd(
            session,
            effective_config,
            reserved_spend_usd=reserved_spend_usd,
        )
        manual_decision = _manual_for_estimate(estimate, remaining_cap_usd=remaining)
        if manual_decision is not None:
            return manual_decision
        return ApprovalDecision(
            approved=True,
            requires_approval=False,
            auto_approved=True,
            estimated_cost_usd=estimate.estimated_cost_usd,
            remaining_cap_usd=remaining,
            billable=estimate.billable,
        )

    return ApprovalDecision(approved=True, requires_approval=False, auto_approved=True)
