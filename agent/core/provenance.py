"""Hub artifact provenance and Trackio metadata helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from agent.core.golden_eval import QUALITY_CONSTRAINTS
from agent.core.post_training_eval import summarize_post_training_eval
from agent.core.redact import scrub, scrub_string
from agent.core.strategy_selector import select_ml_strategy
from agent.core.trainability import assess_trainability

_TRACKIO_DEPENDENCY_RE = re.compile(
    r"^\s*trackio\s*(?:\[[^\]]+\])?\s*(?:$|[<>=!~;])", re.IGNORECASE
)


def build_training_provenance(
    *,
    base_model: str | None = None,
    dataset_profile: Mapping[str, Any] | None = None,
    training_method: str | None = None,
    post_training_eval: Mapping[str, Any] | None = None,
    cost: Mapping[str, Any] | None = None,
    hardware: Mapping[str, Any] | str | None = None,
    timeout: str | None = None,
    limitations: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a scrubbed provenance payload for training or dataset artifacts."""

    profile = dict(dataset_profile or {})
    trainability = _trainability(profile)
    strategy = _strategy(profile, trainability)
    golden_eval = _golden_eval_summary(profile)

    payload = {
        "base_model": base_model,
        "dataset": _dataset_summary(profile),
        "row_count": _row_count(profile),
        "examples_count": _row_count(profile),
        "trainability": trainability,
        "strategy": strategy,
        "golden_eval": golden_eval,
        "post_training_eval": summarize_post_training_eval(post_training_eval),
        "limitations": _limitations(limitations, trainability=trainability),
        "training": {
            "method": training_method,
            "cost": dict(cost or {}),
            "hardware": dict(hardware) if isinstance(hardware, Mapping) else hardware,
            "timeout": timeout,
        },
    }
    return scrub(_drop_empty(payload))


def build_training_job_metadata(args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return scrubbed HF Jobs metadata with Trackio readiness warnings."""

    args = args or {}
    trackio_config = args.get("trackio")
    trackio = dict(trackio_config) if isinstance(trackio_config, Mapping) else {}
    dependency_enabled = _has_trackio_dependency(args.get("dependencies"))
    explicit_enabled = bool(trackio)
    warnings: list[str] = []

    if not dependency_enabled and not explicit_enabled:
        warnings.append("missing_trackio")

    trackio_metadata = {
        "enabled": dependency_enabled or explicit_enabled,
        "dependency": dependency_enabled,
        "project": trackio.get("project"),
        "space_id": trackio.get("space_id"),
        "run_name": trackio.get("run_name"),
        "dashboard_url": trackio.get("dashboard_url"),
    }

    metadata = {
        "trackio": _drop_empty(trackio_metadata),
        "warnings": warnings,
    }
    provenance = args.get("provenance")
    if isinstance(provenance, Mapping):
        metadata["provenance"] = provenance
    return scrub(metadata)


def build_artifact_card(
    provenance: Mapping[str, Any],
    *,
    artifact_type: str = "model",
) -> str:
    """Render a concise model or dataset card section from provenance."""

    safe = scrub(provenance)
    dataset = safe.get("dataset") if isinstance(safe.get("dataset"), Mapping) else {}
    training = safe.get("training") if isinstance(safe.get("training"), Mapping) else {}
    trainability = (
        safe.get("trainability")
        if isinstance(safe.get("trainability"), Mapping)
        else {}
    )
    strategy = safe.get("strategy") if isinstance(safe.get("strategy"), Mapping) else {}
    golden_eval = (
        safe.get("golden_eval") if isinstance(safe.get("golden_eval"), Mapping) else {}
    )
    post_training_eval = (
        safe.get("post_training_eval")
        if isinstance(safe.get("post_training_eval"), Mapping)
        else {}
    )
    limitations = _string_list(safe.get("limitations"))

    title = "Model" if artifact_type == "model" else "Dataset"
    lines = [
        f"# {title} Provenance",
        "",
        "## Source",
        f"- Base model: {_display(safe.get('base_model'))}",
        f"- Dataset: {_display(dataset.get('repo_id') or dataset.get('path'))}",
        f"- Row count: {_display(dataset.get('row_count') or safe.get('row_count'))}",
        "",
        "## Training",
        f"- Method: {_display(training.get('method'))}",
        f"- Hardware: {_display(training.get('hardware'))}",
        f"- Timeout: {_display(training.get('timeout'))}",
        "",
        "## Quality Intelligence",
        f"- Trainability Risk: {_display(trainability.get('risk_level'))}",
        f"- Strategy Recommendation: {_display(strategy.get('strategy'))}",
        f"- Golden Eval: {_display(golden_eval.get('case_count'))} cases",
        f"- Post-Training Eval: {_display(post_training_eval.get('status'))}",
        "",
        "## Limitations",
    ]
    lines.extend(f"- {item}" for item in (limitations or ["Not specified."]))
    return scrub_string("\n".join(lines))


def _dataset_summary(profile: Mapping[str, Any]) -> dict[str, Any]:
    source = profile.get("source") if isinstance(profile.get("source"), Mapping) else {}
    summary = {
        "type": source.get("type"),
        "repo_id": source.get("repo_id") or profile.get("repo_id"),
        "path": source.get("path") or profile.get("path"),
        "format": profile.get("format") or source.get("format"),
        "row_count": _row_count(profile),
        "examples_count": _row_count(profile),
        "profiled_row_count": profile.get("profiled_row_count"),
        "statistics_basis": profile.get("statistics_basis"),
        "inferred_shape": profile.get("inferred_shape"),
        "columns": profile.get("columns"),
    }
    if source.get("split"):
        summary["split"] = source.get("split")
    if source.get("config"):
        summary["config"] = source.get("config")
    return _drop_empty(summary)


def _trainability(profile: Mapping[str, Any]) -> dict[str, Any]:
    embedded = profile.get("trainability")
    if isinstance(embedded, Mapping):
        return dict(embedded)
    return assess_trainability(profile).to_dict()


def _strategy(
    profile: Mapping[str, Any], trainability: Mapping[str, Any]
) -> dict[str, Any]:
    embedded = profile.get("strategy")
    if isinstance(embedded, Mapping):
        return dict(embedded)
    return select_ml_strategy(profile, trainability).to_dict()


def _golden_eval_summary(profile: Mapping[str, Any]) -> dict[str, Any]:
    golden_eval = (
        profile.get("golden_eval")
        if isinstance(profile.get("golden_eval"), Mapping)
        else {}
    )
    cases = (
        golden_eval.get("cases") if isinstance(golden_eval.get("cases"), list) else []
    )
    task_types = sorted(
        {
            str(case.get("task_type"))
            for case in cases
            if isinstance(case, Mapping) and case.get("task_type")
        }
    )
    return _drop_empty(
        {
            "case_count": golden_eval.get("case_count", len(cases)),
            "quality_constraints": golden_eval.get(
                "quality_constraints", list(QUALITY_CONSTRAINTS)
            ),
            "task_types": task_types,
        }
    )


def _limitations(
    explicit: Sequence[str] | None,
    *,
    trainability: Mapping[str, Any],
) -> list[str]:
    if explicit:
        return [str(item) for item in explicit if str(item).strip()]
    reasons = _string_list(trainability.get("reasons"))
    if reasons:
        return reasons
    return ["No limitations were provided."]


def _has_trackio_dependency(dependencies: Any) -> bool:
    if not isinstance(dependencies, Sequence) or isinstance(dependencies, (str, bytes)):
        return False
    return any(
        _TRACKIO_DEPENDENCY_RE.match(str(dependency)) for dependency in dependencies
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _row_count(profile: Mapping[str, Any]) -> Any:
    return (
        profile.get("row_count")
        or profile.get("num_rows")
        or profile.get("examples_count")
    )


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_empty(item)
            for key, item in value.items()
            if item is not None and item != [] and item != {}
        }
    if isinstance(value, list):
        return [_drop_empty(item) for item in value if item is not None]
    return value


def _display(value: Any) -> str:
    if value is None or value == "":
        return "Not specified"
    if isinstance(value, (dict, list)):
        return str(scrub(value))
    return str(value)
