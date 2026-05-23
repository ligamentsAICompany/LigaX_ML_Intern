"""Deterministic structured-table reference lookup for small factual datasets."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from agent.core.golden_eval import NOT_SPECIFIED
from agent.core.redact import scrub, scrub_string

LookupStatus = Literal["answered", "no_answer", "unsupported"]

KEY_COLUMN_CANDIDATES = (
    "Form No",
    "Form",
    "Category",
    "Purpose",
    "Applicable Sections",
    "Section",
    "User Categories",
    "Filing Frequency",
)
ANSWER_COLUMN_CANDIDATES = (
    "Purpose",
    "Form No",
    "Due Dates",
    "Applicable Sections",
    "User Categories",
    "Filing Frequency",
    "Old/New Regime Applicability",
    "Category",
    "API / E-Filing Mapping Possibilities",
    "Description",
    "Limit",
    "Applicability",
    "Rate",
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")
_PARENS = re.compile(r"\([^)]*\)")


@dataclass(frozen=True)
class ReferenceLookupAnswer:
    """Source-grounded answer returned by a structured reference index."""

    status: LookupStatus
    answer: str
    question: str
    source_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceLookupSummary:
    """Readiness summary suitable for dataset inspection payloads."""

    ready: bool
    row_count: int
    key_columns: list[str]
    answer_columns: list[str]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReferenceIndex:
    """Small in-memory lookup index over structured row dictionaries."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        columns: Sequence[str] | None = None,
        source: Mapping[str, Any] | None = None,
    ) -> None:
        self.rows = [dict(row) for row in rows if isinstance(row, Mapping)]
        self.columns = _collect_columns(self.rows, columns)
        self.source = scrub(dict(source or {}))
        self.key_columns = _present_columns(self.columns, KEY_COLUMN_CANDIDATES)
        self.answer_columns = _present_columns(self.columns, ANSWER_COLUMN_CANDIDATES)

    def answer(self, question: str) -> ReferenceLookupAnswer:
        """Answer a question directly from the best matching source row."""

        clean_question = scrub_string(str(question or "").strip())
        answer_column = _infer_answer_column(clean_question, self.columns)
        if answer_column is None:
            return _unsupported(clean_question)

        row_index, row, ambiguity_metadata = self._best_row(
            clean_question, answer_column
        )
        if ambiguity_metadata is not None:
            return ReferenceLookupAnswer(
                status="no_answer",
                answer=NOT_SPECIFIED,
                question=clean_question,
                source_metadata=ambiguity_metadata,
            )
        if row is None or row_index is None:
            return _unsupported(clean_question)

        answer = _clean_value(row.get(answer_column))
        status: LookupStatus = "answered" if answer != NOT_SPECIFIED else "no_answer"
        match_column = _best_match_column(clean_question, row, self.key_columns)
        columns_used = (
            [match_column, answer_column] if match_column else [answer_column]
        )
        source_fields = {
            column: _clean_value(row.get(column))
            for column in columns_used
            if column in row or column == answer_column
        }

        return ReferenceLookupAnswer(
            status=status,
            answer=answer,
            question=clean_question,
            source_metadata={
                "row_index": row_index,
                "source": self.source,
                "columns_used": columns_used,
                "source_fields": scrub(source_fields),
            },
        )

    def summary(self) -> ReferenceLookupSummary:
        ready = bool(self.rows and self.key_columns and self.answer_columns)
        return ReferenceLookupSummary(
            ready=ready,
            row_count=len(self.rows),
            key_columns=list(self.key_columns),
            answer_columns=list(self.answer_columns),
            status="ready" if ready else "insufficient_reference_columns",
        )

    def _best_row(
        self, question: str, answer_column: str
    ) -> tuple[int | None, Mapping[str, Any] | None, dict[str, Any] | None]:
        scored: list[tuple[int, int, Mapping[str, Any]]] = []
        match_columns = [
            column for column in self.key_columns if column != answer_column
        ] or self.key_columns
        for row_index, row in enumerate(self.rows):
            score = max(
                (_score_value(question, row.get(column)) for column in match_columns),
                default=0,
            )
            if score > 0:
                scored.append((score, row_index, row))
        if not scored:
            return None, None, None
        score, row_index, row = max(scored, key=lambda item: (item[0], -item[1]))
        if score < 40:
            return None, None, None
        top_matches = [item for item in scored if item[0] == score]
        if _has_conflicting_answers(top_matches, answer_column):
            return (
                None,
                None,
                {
                    "reason": "ambiguous_top_match",
                    "row_indices": [item[1] for item in top_matches],
                    "answer_column": answer_column,
                    "match_score": score,
                    "source": self.source,
                },
            )
        return row_index, row, None


def build_reference_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    source: Mapping[str, Any] | None = None,
) -> ReferenceIndex:
    """Build a deterministic lookup index from normalized structured rows."""

    return ReferenceIndex(rows, columns=columns, source=source)


def build_reference_lookup_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return lightweight reference lookup readiness metadata for inspection."""

    return (
        build_reference_index(rows, columns=columns, source=source).summary().to_dict()
    )


def _unsupported(question: str) -> ReferenceLookupAnswer:
    return ReferenceLookupAnswer(
        status="unsupported",
        answer=NOT_SPECIFIED,
        question=question,
        source_metadata={},
    )


def _infer_answer_column(question: str, columns: Sequence[str]) -> str | None:
    column_map = {column.lower(): column for column in columns}
    normalized = _normalize(question)

    if "which form" in normalized or "what form" in normalized:
        return column_map.get("form no") or column_map.get("form")
    if "due date" in normalized or "deadline" in normalized or "when" in normalized:
        return column_map.get("due dates")
    if "who" in normalized or "user categor" in normalized or "eligible" in normalized:
        return column_map.get("user categories") or column_map.get("applicability")
    if "section" in normalized:
        return column_map.get("applicable sections") or column_map.get("section")
    if "how often" in normalized or "frequency" in normalized or "filed" in normalized:
        return column_map.get("filing frequency")
    if "regime" in normalized:
        return column_map.get("old/new regime applicability")
    if "category" in normalized or "type of" in normalized:
        return column_map.get("category")
    if "api" in normalized or "e filing" in normalized or "mapping" in normalized:
        return column_map.get("api / e-filing mapping possibilities")
    if "used for" in normalized or "purpose" in normalized:
        return (
            column_map.get("purpose")
            or column_map.get("description")
            or column_map.get("applicability")
        )
    return None


def _best_match_column(
    question: str, row: Mapping[str, Any], key_columns: Sequence[str]
) -> str | None:
    scored = [
        (_score_value(question, row.get(column)), column)
        for column in key_columns
        if column in row
    ]
    if not scored:
        return None
    score, column = max(scored, key=lambda item: item[0])
    return column if score > 0 else None


def _score_value(question: str, value: Any) -> int:
    text = _present_text(value)
    if not text:
        return 0
    question_text = str(question or "").lower()
    normalized_question = _normalize(question)
    compact_question = normalized_question.replace(" ", "")
    question_tokens = set(normalized_question.split())
    best = 0
    for alias in _aliases(text):
        normalized_alias = _normalize(alias)
        if not normalized_alias:
            continue
        compact_alias = normalized_alias.replace(" ", "")
        if _is_identifier_like(compact_alias):
            if _contains_identifier_alias(question_text, compact_alias):
                best = max(best, 130 + len(compact_alias))
            continue
        alias_tokens = set(normalized_alias.split())
        if normalized_alias in normalized_question:
            best = max(best, 120 + len(normalized_alias))
        if compact_alias and compact_alias in compact_question:
            best = max(best, 110 + len(compact_alias))
        overlap = len(alias_tokens & question_tokens)
        if overlap:
            best = max(best, overlap * 25)
    return best


def _has_conflicting_answers(
    matches: Sequence[tuple[int, int, Mapping[str, Any]]], answer_column: str
) -> bool:
    if len(matches) < 2:
        return False
    answers = {_clean_value(row.get(answer_column)) for _, _, row in matches}
    return len(answers) > 1


def _is_identifier_like(compact_alias: str) -> bool:
    return (
        bool(compact_alias)
        and any(char.isalpha() for char in compact_alias)
        and any(char.isdigit() for char in compact_alias)
    )


def _contains_identifier_alias(question: str, compact_alias: str) -> bool:
    pattern = r"(?<![a-z0-9])" + r"[^a-z0-9]*".join(
        re.escape(char) for char in compact_alias
    )
    pattern += r"(?![a-z0-9])"
    return bool(re.search(pattern, question))


def _aliases(value: str) -> set[str]:
    aliases = {value}
    without_parens = _PARENS.sub("", value).strip()
    if without_parens:
        aliases.add(without_parens)
    aliases.add(value.replace("-", " "))
    aliases.add(_LETTER_DIGIT_BOUNDARY.sub(" ", value))
    if value.lower().startswith("form "):
        aliases.add(value[5:])
    return aliases


def _normalize(value: Any) -> str:
    text = _LETTER_DIGIT_BOUNDARY.sub(" ", str(value or "").lower())
    text = _NON_ALNUM.sub(" ", text)
    return " ".join(text.split())


def _clean_value(value: Any) -> str:
    text = _present_text(value)
    return scrub_string(text) if text else NOT_SPECIFIED


def _present_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    return text or None


def _collect_columns(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None
) -> list[str]:
    if columns is not None:
        return [str(column) for column in columns if str(column).strip()]
    collected: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row:
            text = str(column)
            if text not in seen:
                collected.append(text)
                seen.add(text)
    return collected


def _present_columns(columns: Sequence[str], candidates: Sequence[str]) -> list[str]:
    lower_to_original = {column.lower(): column for column in columns}
    return [
        lower_to_original[candidate.lower()]
        for candidate in candidates
        if candidate.lower() in lower_to_original
    ]
