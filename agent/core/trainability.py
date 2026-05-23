"""Deterministic dataset suitability scoring before fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

Recommendation = Literal["fine_tune", "rag", "hybrid", "data_needed"]
RiskLevel = Literal["low", "medium", "high"]

_INSTRUCTION_COLUMN_SETS = (
    frozenset({"messages"}),
    frozenset({"prompt", "completion"}),
    frozenset({"instruction", "output"}),
    frozenset({"instruction", "response"}),
    frozenset({"input", "output"}),
    frozenset({"prompt", "chosen", "rejected"}),
)
_REFERENCE_COLUMN_HINTS = {
    "form no",
    "purpose",
    "section",
    "applicable sections",
    "description",
    "limit",
    "applicability",
    "rule",
    "rate",
    "category",
    "deduction",
    "due dates",
    "filing frequency",
    "user categories",
    "old/new regime applicability",
}
_TRAINING_FORMATS = {"jsonl", "json", "parquet"}
_STRUCTURED_FORMATS = {"csv", "xlsx", "xls"}
_DOCUMENT_FORMATS = {"pdf", "docx"}


@dataclass(frozen=True)
class TrainabilityResult:
    """Suitability score and decision for a candidate training dataset."""

    score: int
    recommendation: Recommendation
    risk_level: RiskLevel
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_trainability(profile: Mapping[str, Any]) -> TrainabilityResult:
    """Score whether a dataset should be directly fine-tuned.

    The input is intentionally a lightweight mapping so callers can pass data
    from local upload validation, HF Dataset Viewer inspection, or future richer
    profiling without depending on one profile class.
    """

    reasons: list[str] = []
    score = 100

    row_count = _coerce_non_negative_int(profile.get("row_count"))
    columns = _normalize_columns(profile.get("columns"))
    dataset_format = _normalize_format(
        profile.get("format") or profile.get("file_type")
    )
    raw_sample_rows = profile.get("sample_rows")
    sample_rows_supplied = isinstance(raw_sample_rows, list)
    sample_rows = _normalize_sample_rows(raw_sample_rows)

    if row_count == 0:
        reasons.append("Dataset has no rows, so there is nothing reliable to train on.")
        score -= 70
    elif row_count is None:
        reasons.append("Row count is unavailable; suitability cannot be confirmed.")
        score -= 20
    elif row_count < 50:
        reasons.append(
            f"Dataset is tiny ({row_count} rows), which is high risk for fine-tuning."
        )
        score -= 40
    elif row_count < 200:
        reasons.append(f"Dataset has only {row_count} rows; fine-tuning may overfit.")
        score -= 25
    elif row_count < 1000:
        reasons.append(
            f"Dataset has {row_count} rows; use a narrow fine-tune or add evaluation coverage."
        )
        score -= 10

    has_instruction_columns = _has_instruction_columns(columns)
    instruction_like = _has_instruction_schema(
        columns,
        sample_rows,
        sample_rows_supplied=sample_rows_supplied,
    )
    structured_reference = _is_structured_reference_table(
        columns=columns,
        dataset_format=dataset_format,
        sample_rows=sample_rows,
        row_count=row_count,
    )
    document_corpus = _is_document_corpus(
        columns=columns,
        dataset_format=dataset_format,
        sample_rows=sample_rows,
    )

    if instruction_like:
        reasons.append(
            "Dataset has instruction/SFT-style fields suitable for supervised fine-tuning."
        )
        if has_instruction_columns and not sample_rows_supplied:
            reasons.append(
                "Sample rows are unavailable, so schema suitability cannot be fully confirmed."
            )
            score -= 25
    else:
        if has_instruction_columns and sample_rows_supplied:
            reasons.append(
                "Sample rows do not contain usable non-empty instruction examples."
            )
        else:
            reasons.append(
                "Dataset does not expose a clear messages or prompt/completion training schema."
            )
        score -= 25

    if structured_reference:
        reasons.append(
            "Dataset looks like a structured reference table, better suited to retrieval or hybrid grounding."
        )
        score -= 30

    if document_corpus:
        reasons.append(
            "Dataset is an extracted document corpus, better suited to retrieval or reference lookup."
        )
        score -= 35

    if dataset_format:
        if dataset_format in _TRAINING_FORMATS:
            reasons.append(
                f"Dataset format ({dataset_format}) is compatible with common HF training loaders."
            )
        elif dataset_format in _STRUCTURED_FORMATS:
            reasons.append(
                f"Dataset format ({dataset_format}) is tabular and needs task conversion before SFT."
            )
            score -= 10
        elif dataset_format in _DOCUMENT_FORMATS:
            reasons.append(
                f"Dataset format ({dataset_format}) is document text and should be chunked for retrieval."
            )
            score -= 10
        else:
            reasons.append(
                f"Dataset format ({dataset_format}) is not a standard fine-tuning signal."
            )
            score -= 10

    missing_fraction = _coerce_fraction(profile.get("missing_fraction"))
    duplicate_fraction = _coerce_fraction(profile.get("duplicate_fraction"))
    if profile.get("statistics_basis") == "sample_rows":
        reasons.append(
            "Missingness and duplicate rates are estimated from sample rows, not the full split."
        )
    if missing_fraction is not None and missing_fraction >= 0.30:
        reasons.append(f"Dataset has high missingness ({missing_fraction:.0%}).")
        score -= 20
    elif missing_fraction is not None and missing_fraction >= 0.10:
        reasons.append(f"Dataset has noticeable missingness ({missing_fraction:.0%}).")
        score -= 10

    if duplicate_fraction is not None and duplicate_fraction >= 0.25:
        reasons.append(f"Dataset has high duplicate rate ({duplicate_fraction:.0%}).")
        score -= 15
    elif duplicate_fraction is not None and duplicate_fraction >= 0.10:
        reasons.append(
            f"Dataset has noticeable duplicate rate ({duplicate_fraction:.0%})."
        )
        score -= 8

    score = max(0, min(100, score))
    recommendation = _recommend(
        score=score,
        row_count=row_count,
        instruction_like=instruction_like,
        structured_reference=structured_reference,
        document_corpus=document_corpus,
        missing_fraction=missing_fraction,
    )
    risk_level = _risk_level(score, recommendation)

    return TrainabilityResult(
        score=score,
        recommendation=recommendation,
        risk_level=risk_level,
        reasons=reasons,
    )


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


def _normalize_format(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().lstrip(".")
    return normalized or None


def _normalize_sample_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _has_instruction_columns(columns: set[str]) -> bool:
    return any(required.issubset(columns) for required in _INSTRUCTION_COLUMN_SETS)


def _has_instruction_schema(
    columns: set[str],
    sample_rows: list[Mapping[str, Any]],
    *,
    sample_rows_supplied: bool = False,
) -> bool:
    for required in _INSTRUCTION_COLUMN_SETS:
        if not required.issubset(columns):
            continue
        if not sample_rows_supplied:
            return True
        if required == {"messages"}:
            return _messages_rows_have_user_and_assistant(sample_rows)
        return _rows_have_non_empty_fields(sample_rows, required)
    return False


def _messages_rows_have_user_and_assistant(
    sample_rows: list[Mapping[str, Any]],
) -> bool:
    for row in sample_rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        roles = set()
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            content = message.get("content")
            if isinstance(role, str) and _non_empty_text(content):
                roles.add(role)
        if {"user", "assistant"}.issubset(roles):
            return True
    return False


def _rows_have_non_empty_fields(
    sample_rows: list[Mapping[str, Any]], required: frozenset[str]
) -> bool:
    for row in sample_rows:
        if all(_non_empty_text(row.get(column)) for column in required):
            return True
    return False


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_structured_reference_table(
    *,
    columns: set[str],
    dataset_format: str | None,
    sample_rows: list[Mapping[str, Any]],
    row_count: int | None,
) -> bool:
    if not columns:
        return False
    if _has_instruction_schema(columns, sample_rows):
        return False

    reference_hint_count = len(columns & _REFERENCE_COLUMN_HINTS)
    compact_tabular = dataset_format in _STRUCTURED_FORMATS and len(columns) >= 3
    tiny_lookup = (
        row_count is not None and row_count < 100 and reference_hint_count >= 2
    )
    scalar_samples = bool(sample_rows) and all(
        not isinstance(value, (list, dict))
        for row in sample_rows
        for value in row.values()
    )
    return tiny_lookup or (
        compact_tabular and reference_hint_count >= 2 and scalar_samples
    )


def _is_document_corpus(
    *,
    columns: set[str],
    dataset_format: str | None,
    sample_rows: list[Mapping[str, Any]],
) -> bool:
    if dataset_format in _DOCUMENT_FORMATS:
        return True
    if "text" not in columns:
        return False
    formats = {
        str(row.get("source_format") or "").strip().lower()
        for row in sample_rows
        if isinstance(row, Mapping)
    }
    return bool(formats & _DOCUMENT_FORMATS)


def _recommend(
    *,
    score: int,
    row_count: int | None,
    instruction_like: bool,
    structured_reference: bool,
    document_corpus: bool,
    missing_fraction: float | None,
) -> Recommendation:
    if row_count == 0 or (missing_fraction is not None and missing_fraction >= 0.40):
        return "data_needed"
    if document_corpus:
        return "rag"
    if structured_reference:
        return "rag" if row_count is not None and row_count < 200 else "hybrid"
    if score < 35:
        return "data_needed"
    if instruction_like and score >= 55:
        return "fine_tune"
    if instruction_like:
        return "data_needed"
    return "hybrid" if score >= 50 else "data_needed"


def _risk_level(score: int, recommendation: Recommendation) -> RiskLevel:
    if recommendation == "data_needed" or score < 50:
        return "high"
    if score < 80 or recommendation in {"rag", "hybrid"}:
        return "medium"
    return "low"
