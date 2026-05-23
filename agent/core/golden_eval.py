"""Deterministic golden eval case generation from inspected dataset samples."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from agent.core.redact import scrub, scrub_string

NOT_SPECIFIED = "Not specified in the source table"
MAX_TEXT_LEN = 240
DEFAULT_MAX_CASES = 25

QUALITY_CONSTRAINTS = [
    "Do not hallucinate values",
    "Expected answer must be concise",
    "Answer must be supported by source fields",
]

_FORM_COLUMN = "Form No"
_PURPOSE_COLUMN = "Purpose"
_DUE_DATE_COLUMN = "Due Dates"
_REGIME_COLUMN = "Old/New Regime Applicability"
_STRUCTURED_SHAPES = {"structured_reference_table", "qa", "unknown"}
_SFT_SHAPES = {"sft_messages", "prompt_completion"}
_SFT_REFERENCE_TASK_TYPE = "sft_reference_eval"
_SFT_HOLDOUT_TASK_TYPE = "sft_holdout"
_SFT_REFERENCE_HOLDOUT_STATUS = "sample_based/unverified"
_SFT_EXPLICIT_HOLDOUT_STATUS = "explicit_holdout"
_HOLDOUT_SPLIT_NAMES = {"eval", "evaluation", "validation", "valid", "test", "holdout"}


@dataclass(frozen=True)
class GoldenEvalCase:
    """One source-grounded evaluation example for pre/post-training checks."""

    id: str
    question: str
    expected_answer: str
    rubric: list[str]
    source_metadata: dict[str, Any]
    task_type: str
    domain: str | None
    quality_constraints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_golden_eval_cases(
    *,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
    dataset_shape: str | None = None,
    source: Mapping[str, Any] | None = None,
    domain: str | None = None,
    max_cases: int = DEFAULT_MAX_CASES,
) -> list[GoldenEvalCase]:
    """Generate deterministic golden eval cases without inventing source facts."""

    normalized_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    normalized_columns = [
        str(column) for column in columns or _collect_columns(normalized_rows)
    ]
    shape = str(dataset_shape or "").strip().lower()
    inferred_domain = domain or _infer_domain(normalized_columns, source)
    limit = max(0, max_cases)

    if shape in _SFT_SHAPES or _has_sft_columns(normalized_columns):
        cases = _sft_cases(
            rows=normalized_rows,
            columns=normalized_columns,
            source=source,
            domain=inferred_domain,
        )
    elif shape in _STRUCTURED_SHAPES or normalized_columns:
        cases = _structured_reference_cases(
            rows=normalized_rows,
            columns=normalized_columns,
            source=source,
            domain=inferred_domain,
        )
    else:
        cases = []

    return cases[:limit]


def _sft_cases(
    *,
    rows: list[dict[str, Any]],
    columns: list[str],
    source: Mapping[str, Any] | None,
    domain: str | None,
) -> list[GoldenEvalCase]:
    cases: list[GoldenEvalCase] = []
    lower_columns = {column.lower(): column for column in columns}
    task_type, holdout_status = _sft_provenance(source)

    for row_index, row in enumerate(rows):
        if "messages" in lower_columns:
            case = _messages_case(
                row=row,
                row_index=row_index,
                messages_column=lower_columns["messages"],
                source=source,
                domain=domain,
                task_type=task_type,
                holdout_status=holdout_status,
            )
            if case is not None:
                cases.append(case)
            continue

        prompt_column = lower_columns.get("prompt") or lower_columns.get("instruction")
        completion_column = (
            lower_columns.get("completion")
            or lower_columns.get("output")
            or lower_columns.get("response")
        )
        if prompt_column and completion_column:
            question = _clean_text(row.get(prompt_column), missing_fallback="")
            answer = _clean_text(row.get(completion_column), missing_fallback="")
            if question and answer:
                cases.append(
                    _case(
                        row_index=row_index,
                        question=question,
                        expected_answer=answer,
                        columns_used=[prompt_column, completion_column],
                        source_fields={
                            prompt_column: question,
                            completion_column: answer,
                        },
                        source=source,
                        domain=domain,
                        task_type=task_type,
                        case_kind="prompt_completion",
                        extra_metadata={"holdout_status": holdout_status},
                    )
                )

    return cases


def _messages_case(
    *,
    row: Mapping[str, Any],
    row_index: int,
    messages_column: str,
    source: Mapping[str, Any] | None,
    domain: str | None,
    task_type: str,
    holdout_status: str,
) -> GoldenEvalCase | None:
    messages = _coerce_messages(row.get(messages_column))
    if not messages:
        return None

    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = _clean_text(message.get("content"), missing_fallback="")
        if role in {"system", "user", "assistant"} and content:
            normalized_messages.append({"role": str(role), "content": content})

    user_message: dict[str, str] | None = None
    assistant_message: dict[str, str] | None = None
    assistant_index: int | None = None
    for index in range(len(normalized_messages) - 1, -1, -1):
        message = normalized_messages[index]
        if message["role"] != "assistant":
            continue
        for previous_index in range(index - 1, -1, -1):
            previous_message = normalized_messages[previous_index]
            if previous_message["role"] == "user":
                user_message = previous_message
                assistant_message = message
                assistant_index = index
                break
        if user_message is not None:
            break

    if user_message is None or assistant_message is None:
        return None

    source_messages = [
        message
        for message in normalized_messages[: assistant_index + 1]
        if message["role"] in {"system", "user"} or message is assistant_message
    ]

    return _case(
        row_index=row_index,
        question=user_message["content"],
        expected_answer=assistant_message["content"],
        columns_used=[messages_column],
        source_fields={messages_column: source_messages},
        source=source,
        domain=domain,
        task_type=task_type,
        case_kind="messages",
        extra_metadata={"holdout_status": holdout_status},
    )


def _structured_reference_cases(
    *,
    rows: list[dict[str, Any]],
    columns: list[str],
    source: Mapping[str, Any] | None,
    domain: str | None,
) -> list[GoldenEvalCase]:
    cases: list[GoldenEvalCase] = []

    if _FORM_COLUMN in columns and _PURPOSE_COLUMN in columns:
        for row_index, row in enumerate(rows):
            form = _present_text(row.get(_FORM_COLUMN))
            purpose = _present_text(row.get(_PURPOSE_COLUMN))
            if form and purpose:
                cases.append(
                    _case(
                        row_index=row_index,
                        question=f"What is {form} used for?",
                        expected_answer=purpose,
                        columns_used=[_FORM_COLUMN, _PURPOSE_COLUMN],
                        source_fields={_FORM_COLUMN: form, _PURPOSE_COLUMN: purpose},
                        source=source,
                        domain=domain,
                        task_type="structured_reference_qa",
                        case_kind="purpose",
                    )
                )
                cases.append(
                    _case(
                        row_index=row_index,
                        question=f"Which form is used for {purpose}?",
                        expected_answer=form,
                        columns_used=[_PURPOSE_COLUMN, _FORM_COLUMN],
                        source_fields={_PURPOSE_COLUMN: purpose, _FORM_COLUMN: form},
                        source=source,
                        domain=domain,
                        task_type="structured_reference_qa",
                        case_kind="reverse_lookup",
                    )
                )
            if form and _DUE_DATE_COLUMN in columns:
                due_date = _clean_text(row.get(_DUE_DATE_COLUMN))
                cases.append(
                    _case(
                        row_index=row_index,
                        question=f"What due date is listed for {form}?",
                        expected_answer=due_date,
                        columns_used=[_FORM_COLUMN, _DUE_DATE_COLUMN],
                        source_fields={_FORM_COLUMN: form, _DUE_DATE_COLUMN: due_date},
                        source=source,
                        domain=domain,
                        task_type="structured_reference_qa",
                        case_kind="due_date",
                    )
                )
            if form and _REGIME_COLUMN in columns:
                regime = _clean_text(row.get(_REGIME_COLUMN))
                cases.append(
                    _case(
                        row_index=row_index,
                        question=f"What regime applicability is listed for {form}?",
                        expected_answer=regime,
                        columns_used=[_FORM_COLUMN, _REGIME_COLUMN],
                        source_fields={_FORM_COLUMN: form, _REGIME_COLUMN: regime},
                        source=source,
                        domain=domain,
                        task_type="structured_reference_qa",
                        case_kind="regime",
                    )
                )
        return cases

    key_column = _first_present(
        columns, ("Section", "Category", "Rule", "Code", "Name")
    )
    answer_column = _first_present(
        columns,
        ("Description", "Purpose", "Limit", "Applicability", "Rule", "Rate"),
        exclude={key_column} if key_column else set(),
    )
    if key_column is None or answer_column is None:
        return cases

    for row_index, row in enumerate(rows):
        key = _present_text(row.get(key_column))
        answer = _present_text(row.get(answer_column))
        if key is None or answer is None:
            continue
        cases.append(
            _case(
                row_index=row_index,
                question=f"What {answer_column.lower()} is listed for {key}?",
                expected_answer=answer,
                columns_used=[key_column, answer_column],
                source_fields={key_column: key, answer_column: answer},
                source=source,
                domain=domain,
                task_type="structured_reference_qa",
                case_kind="lookup",
            )
        )

    return cases


def _case(
    *,
    row_index: int,
    question: str,
    expected_answer: str,
    columns_used: list[str],
    source_fields: Mapping[str, Any],
    source: Mapping[str, Any] | None,
    domain: str | None,
    task_type: str,
    case_kind: str,
    extra_metadata: Mapping[str, Any] | None = None,
) -> GoldenEvalCase:
    source_metadata = {
        "row_index": row_index,
        "source": scrub(dict(source or {})),
        "columns_used": columns_used,
        "source_fields": scrub(dict(source_fields)),
    }
    if extra_metadata:
        source_metadata.update(scrub(dict(extra_metadata)))
    return GoldenEvalCase(
        id=f"{task_type}:{row_index}:{case_kind}",
        question=_truncate(scrub_string(question)),
        expected_answer=_truncate(scrub_string(expected_answer)),
        rubric=[
            "Answer must be supported by the source fields.",
            "Do not add values that are missing from the source.",
            f"Use {NOT_SPECIFIED!r} when the requested field is empty.",
        ],
        source_metadata=source_metadata,
        task_type=task_type,
        domain=domain,
        quality_constraints=list(QUALITY_CONSTRAINTS),
    )


def _clean_text(value: Any, *, missing_fallback: str = NOT_SPECIFIED) -> str:
    if value is None:
        return missing_fallback
    if isinstance(value, float) and value != value:
        return missing_fallback
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    else:
        text = str(value)
    text = scrub_string(text.strip())
    if not text:
        return missing_fallback
    return _truncate(text)


def _present_text(value: Any) -> str | None:
    text = _clean_text(value, missing_fallback="")
    return text or None


def _truncate(text: str) -> str:
    if len(text) <= MAX_TEXT_LEN:
        return text
    return text[:MAX_TEXT_LEN].rstrip() + "..."


def _coerce_messages(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [message for message in value if isinstance(message, Mapping)]


def _collect_columns(rows: list[Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row:
            text = str(column)
            if text not in seen:
                columns.append(text)
                seen.add(text)
    return columns


def _has_sft_columns(columns: list[str]) -> bool:
    lowered = {column.lower() for column in columns}
    return (
        "messages" in lowered
        or {"prompt", "completion"}.issubset(lowered)
        or {"instruction", "output"}.issubset(lowered)
        or {"instruction", "response"}.issubset(lowered)
    )


def _infer_domain(
    columns: Sequence[str], source: Mapping[str, Any] | None
) -> str | None:
    column_set = {column.lower() for column in columns}
    source_text = json.dumps(dict(source or {}), sort_keys=True).lower()
    if {
        "form no",
        "old/new regime applicability",
    } & column_set or "income_tax" in source_text:
        return "income_tax"
    if "tax" in source_text:
        return "income_tax"
    return None


def _sft_provenance(source: Mapping[str, Any] | None) -> tuple[str, str]:
    if _source_indicates_holdout(source):
        return _SFT_HOLDOUT_TASK_TYPE, _SFT_EXPLICIT_HOLDOUT_STATUS
    return _SFT_REFERENCE_TASK_TYPE, _SFT_REFERENCE_HOLDOUT_STATUS


def _source_indicates_holdout(source: Mapping[str, Any] | None) -> bool:
    if not source:
        return False
    for key, value in source.items():
        key_text = str(key).lower()
        if not any(token in key_text for token in ("split", "source", "path", "name")):
            continue
        value_text = str(value).lower().replace("\\", "/")
        parts = {
            part
            for separator in ("/", "\\", "_", "-", ".", " ")
            for part in value_text.split(separator)
        }
        if _HOLDOUT_SPLIT_NAMES & parts:
            return True
    return False


def _first_present(
    columns: Sequence[str],
    candidates: Sequence[str],
    *,
    exclude: set[str | None] | None = None,
) -> str | None:
    excluded = exclude or set()
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        column = lower_to_original.get(candidate.lower())
        if column and column not in excluded:
            return column
    return None
