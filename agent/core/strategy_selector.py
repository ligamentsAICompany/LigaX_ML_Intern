"""Product-facing ML strategy selection from dataset quality signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from agent.core.trainability import assess_trainability

Strategy = Literal["fine_tune", "rag", "hybrid", "data_needed"]
RiskLevel = Literal["low", "medium", "high"]
MethodHint = Literal["sft", "dpo", "grpo"]


@dataclass(frozen=True)
class StrategyDecision:
    """Recommendation for the next ML approach before any training starts."""

    strategy: Strategy
    confidence: float
    risk_level: RiskLevel
    reasons: list[str]
    required_next_actions: list[str]
    can_train_without_override: bool
    requires_user_override_for_training: bool
    method_hint: MethodHint | None = None
    override_message: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_ml_strategy(
    profile: Mapping[str, Any],
    trainability_result: Mapping[str, Any] | None = None,
) -> StrategyDecision:
    """Choose the product-facing ML strategy for a profiled dataset."""

    trainability = _trainability(profile, trainability_result)
    row_count = _coerce_non_negative_int(profile.get("row_count"))
    missing_fraction = _coerce_fraction(profile.get("missing_fraction"))
    shape = str(profile.get("inferred_shape") or "").lower()
    columns = _normalize_columns(profile.get("columns"))
    trainability_recommendation = _strategy(trainability.get("recommendation"))
    trainability_risk = _risk_level(trainability.get("risk_level"))
    trainability_score = _coerce_score(trainability.get("score"))

    method_hint = _method_hint(shape, columns)
    structured_reference = _is_structured_reference(
        shape=shape,
        trainability=trainability,
    )
    document_corpus = shape == "document_corpus"
    dirty_or_empty = _is_dirty_or_empty(
        row_count=row_count,
        missing_fraction=missing_fraction,
        trainability=trainability,
        shape=shape,
    )

    if dirty_or_empty:
        return _data_needed_decision(
            trainability=trainability,
            confidence=0.90,
            reasons=[
                "Dataset quality is too weak or unknown for a reliable ML strategy.",
            ],
        )

    if document_corpus:
        return StrategyDecision(
            strategy="rag",
            confidence=0.88,
            risk_level="medium",
            reasons=_merge_reasons(
                [
                    "Dataset is extracted document text and should be grounded through retrieval.",
                    "Direct fine-tuning is only appropriate if explicit instruction/answer pairs are present.",
                ],
                trainability,
            ),
            required_next_actions=[
                "Build a retrieval index over document chunks before answering factual questions.",
                "Create grounded evaluation questions that cite the uploaded documents.",
            ],
            can_train_without_override=False,
            requires_user_override_for_training=True,
            override_message=(
                "Direct fine-tuning requires user override because uploaded documents are safer as retrieval context."
            ),
            metadata=_metadata(trainability),
        )

    if structured_reference:
        strategy: Strategy = (
            "rag" if row_count is not None and row_count < 200 else "hybrid"
        )
        return _reference_decision(
            strategy=strategy,
            row_count=row_count,
            trainability=trainability,
        )

    if method_hint and _method_hint_blocked_by_trainability(
        profile=profile,
        trainability=trainability,
        shape=shape,
    ):
        return _data_needed_decision(
            trainability=trainability,
            confidence=0.84,
            reasons=[
                "Column-level method hints cannot override failed sample validation.",
            ],
        )

    if method_hint == "grpo":
        return _fine_tune_decision(
            method_hint="grpo",
            risk_level=_higher_risk(trainability_risk, "medium"),
            confidence=_confidence(trainability_score, floor=0.65, ceiling=0.82),
            trainability=trainability,
            reasons=[
                "Dataset has prompt-only examples that fit a reward-guided training workflow.",
            ],
            required_next_actions=[
                "Define the reward function or verifiable objective before launching GRPO.",
                "Run a small evaluation harness before scaling the training job.",
            ],
            requires_override=True,
            override_message=(
                "Training requires user override until a reward function or verifiable "
                "objective is supplied and validated for GRPO."
            ),
            override_actions=[
                "Get explicit user override or validate the reward objective before launching GRPO.",
            ],
        )

    if method_hint in {"sft", "dpo"} or trainability_recommendation == "fine_tune":
        risk_level = _fine_tune_risk(
            row_count=row_count,
            trainability_risk=trainability_risk,
        )
        return _fine_tune_decision(
            method_hint=method_hint or "sft",
            risk_level=risk_level,
            confidence=_fine_tune_confidence(
                row_count=row_count,
                risk_level=risk_level,
                trainability_score=trainability_score,
            ),
            trainability=trainability,
            reasons=_fine_tune_reasons(method_hint or "sft", row_count),
            required_next_actions=_fine_tune_actions(method_hint or "sft"),
        )

    if trainability_recommendation in {"rag", "hybrid"}:
        return _reference_decision(
            strategy=trainability_recommendation,
            row_count=row_count,
            trainability=trainability,
        )

    return _data_needed_decision(
        trainability=trainability,
        confidence=0.80,
        reasons=[
            "Dataset shape is not specific enough to choose fine-tuning or retrieval.",
        ],
    )


def _trainability(
    profile: Mapping[str, Any],
    explicit: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if explicit is not None:
        return explicit
    embedded = profile.get("trainability")
    if isinstance(embedded, Mapping):
        return embedded
    return assess_trainability(profile).to_dict()


def _data_needed_decision(
    *,
    trainability: Mapping[str, Any],
    confidence: float,
    reasons: list[str],
) -> StrategyDecision:
    return StrategyDecision(
        strategy="data_needed",
        confidence=confidence,
        risk_level="high",
        reasons=_merge_reasons(reasons, trainability),
        required_next_actions=[
            "Collect, clean, or relabel data until the profile has usable training or retrieval signals.",
            "Re-run dataset inspection before approving any training job.",
        ],
        can_train_without_override=False,
        requires_user_override_for_training=True,
        override_message=(
            "Training requires user override because the selector needs more or cleaner data."
        ),
        metadata=_metadata(trainability),
    )


def _reference_decision(
    *,
    strategy: Strategy,
    row_count: int | None,
    trainability: Mapping[str, Any],
) -> StrategyDecision:
    tiny_text = (
        f"Dataset has only {row_count} rows and looks like a reference table."
        if row_count is not None
        else "Dataset looks like a reference table."
    )
    action = (
        "Build a retrieval index and answer from grounded rows before considering fine-tuning."
        if strategy == "rag"
        else "Use retrieval for factual grounding, then fine-tune only if behavior gaps remain."
    )
    return StrategyDecision(
        strategy=strategy,
        confidence=0.86 if strategy == "rag" else 0.78,
        risk_level="high"
        if _risk_level(trainability.get("risk_level")) == "high"
        else "medium",
        reasons=_merge_reasons(
            [
                tiny_text,
                "Tiny factual/reference tables should not be blindly fine-tuned.",
            ],
            trainability,
        ),
        required_next_actions=[
            action,
            "Create evaluation questions that verify answers are grounded in the reference data.",
        ],
        can_train_without_override=False,
        requires_user_override_for_training=True,
        override_message=(
            "Direct fine-tuning requires user override because retrieval or hybrid grounding is safer."
        ),
        metadata=_metadata(trainability),
    )


def _fine_tune_decision(
    *,
    method_hint: MethodHint,
    risk_level: RiskLevel,
    confidence: float,
    trainability: Mapping[str, Any],
    reasons: list[str],
    required_next_actions: list[str],
    requires_override: bool | None = None,
    override_message: str | None = None,
    override_actions: list[str] | None = None,
) -> StrategyDecision:
    high_risk_override = risk_level == "high"
    requires_override = (
        high_risk_override if requires_override is None else requires_override
    )
    requires_override = requires_override or high_risk_override
    method_name = {
        "sft": "supervised fine-tuning",
        "dpo": "DPO preference fine-tuning",
        "grpo": "GRPO-style reward optimization",
    }[method_hint]
    extra_actions = []
    if high_risk_override:
        extra_actions.append(
            "Get explicit user override before launching this high-risk fine-tune."
        )
    if override_actions:
        extra_actions.extend(override_actions)
    high_risk_message = (
        "User override required for this high-risk fine-tune; the dataset is too small "
        "or risky for automatic approval."
    )
    return StrategyDecision(
        strategy="fine_tune",
        confidence=confidence,
        risk_level=risk_level,
        reasons=_merge_reasons(
            [f"Recommended method: {method_name}.", *reasons], trainability
        ),
        required_next_actions=required_next_actions + extra_actions,
        can_train_without_override=not requires_override,
        requires_user_override_for_training=requires_override,
        method_hint=method_hint,
        override_message=override_message
        or (high_risk_message if high_risk_override else ""),
        metadata=_metadata(trainability, method_hint=method_hint),
    )


def _fine_tune_reasons(method_hint: MethodHint, row_count: int | None) -> list[str]:
    if method_hint == "dpo":
        return ["Dataset has prompt/chosen/rejected preference fields."]
    if row_count is not None and row_count < 50:
        return [
            "Dataset is fine-tune shaped but extremely small, so overfitting risk is high."
        ]
    return ["Dataset has valid instruction/SFT-style fields for training."]


def _fine_tune_actions(method_hint: MethodHint) -> list[str]:
    if method_hint == "dpo":
        return [
            "Use a DPO-capable trainer and keep prompt/chosen/rejected columns intact.",
            "Evaluate preference alignment on held-out prompts before scaling.",
        ]
    if method_hint == "grpo":
        return [
            "Define the reward function or verifiable objective before launching GRPO.",
            "Run a small evaluation harness before scaling the training job.",
        ]
    return [
        "Run a small SFT job with held-out evaluation before scaling.",
        "Track quality and overfitting metrics during training.",
    ]


def _metadata(
    trainability: Mapping[str, Any],
    *,
    method_hint: MethodHint | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"trainability": dict(trainability)}
    if method_hint:
        metadata["method_recommendation"] = method_hint
    return metadata


def _merge_reasons(reasons: list[str], trainability: Mapping[str, Any]) -> list[str]:
    merged = list(reasons)
    for reason in trainability.get("reasons") or []:
        text = str(reason)
        if text and text not in merged:
            merged.append(text)
    return merged


def _method_hint(shape: str, columns: set[str]) -> MethodHint | None:
    if shape == "dpo" or {"prompt", "chosen", "rejected"}.issubset(columns):
        return "dpo"
    if shape == "grpo_prompt_only" or columns == {"prompt"}:
        return "grpo"
    if shape in {"sft_messages", "prompt_completion"}:
        return "sft"
    if {"messages"}.issubset(columns) or {"prompt", "completion"}.issubset(columns):
        return "sft"
    if {"instruction", "output"}.issubset(columns) or {
        "instruction",
        "response",
    }.issubset(columns):
        return "sft"
    return None


def _method_hint_blocked_by_trainability(
    *,
    profile: Mapping[str, Any],
    trainability: Mapping[str, Any],
    shape: str,
) -> bool:
    recommendation = _strategy(trainability.get("recommendation"))
    if recommendation == "data_needed":
        return True
    if (
        _risk_level(trainability.get("risk_level")) == "high"
        and recommendation != "fine_tune"
    ):
        return True

    reason_text = " ".join(
        str(reason).lower() for reason in trainability.get("reasons") or []
    )
    if "sample rows do not contain usable" in reason_text:
        return True
    if "sample rows are unavailable" in reason_text:
        return True

    sample_rows = profile.get("sample_rows")
    validated_shapes = {"dpo", "grpo_prompt_only", "sft_messages", "prompt_completion"}
    return (
        isinstance(sample_rows, list)
        and not sample_rows
        and shape not in validated_shapes
    )


def _is_structured_reference(
    *,
    shape: str,
    trainability: Mapping[str, Any],
) -> bool:
    if shape == "structured_reference_table":
        return True
    recommendation = _strategy(trainability.get("recommendation"))
    reason_text = " ".join(
        str(reason).lower() for reason in trainability.get("reasons") or []
    )
    return recommendation in {"rag", "hybrid"} and "structured reference" in reason_text


def _is_dirty_or_empty(
    *,
    row_count: int | None,
    missing_fraction: float | None,
    trainability: Mapping[str, Any],
    shape: str,
) -> bool:
    if row_count == 0:
        return True
    if missing_fraction is not None and missing_fraction >= 0.40:
        return True
    return (
        shape in {"", "unknown"}
        and _strategy(trainability.get("recommendation")) == "data_needed"
        and _risk_level(trainability.get("risk_level")) == "high"
    )


def _fine_tune_risk(
    *,
    row_count: int | None,
    trainability_risk: RiskLevel,
) -> RiskLevel:
    if row_count is not None and row_count < 50:
        return "high"
    return trainability_risk


def _fine_tune_confidence(
    *,
    row_count: int | None,
    risk_level: RiskLevel,
    trainability_score: int,
) -> float:
    if risk_level == "high":
        return 0.42
    if row_count is not None and row_count >= 1000:
        return _confidence(trainability_score, floor=0.86, ceiling=0.96)
    return _confidence(trainability_score, floor=0.65, ceiling=0.84)


def _confidence(score: int, *, floor: float, ceiling: float) -> float:
    normalized = max(0.0, min(1.0, score / 100))
    return round(max(floor, min(ceiling, normalized)), 2)


def _higher_risk(left: RiskLevel, right: RiskLevel) -> RiskLevel:
    order = {"low": 0, "medium": 1, "high": 2}
    return left if order[left] >= order[right] else right


def _strategy(value: Any) -> Strategy | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"fine_tune", "rag", "hybrid", "data_needed"}:
        return normalized  # type: ignore[return-value]
    return None


def _risk_level(value: Any) -> RiskLevel:
    normalized = str(value or "").strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized  # type: ignore[return-value]
    return "high"


def _coerce_score(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))


def _coerce_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, integer)


def _coerce_fraction(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if fraction < 0:
        return None
    return min(1.0, fraction)


def _normalize_columns(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(column).strip().lower() for column in value if str(column).strip()}
