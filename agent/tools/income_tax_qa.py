"""Deterministic Income Tax master-table to QA dataset conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SOURCE_COLUMNS = (
    "Form No",
    "Category",
    "Purpose",
    "Applicable Sections",
    "Filing Frequency",
    "Due Dates",
    "User Categories",
    "Old/New Regime Applicability",
    "API / E-Filing Mapping Possibilities",
)

SYSTEM_PROMPT = (
    "You are an Income Tax form assistant. Answer only from the provided "
    "Income Tax Master reference row. If a field is missing, say it is not "
    "specified in the source table."
)


@dataclass(frozen=True)
class TaxFormRecord:
    """Normalized representation of one workbook row."""

    form_no: str
    category: str
    purpose: str
    applicable_sections: str
    filing_frequency: str
    due_dates: str
    user_categories: str
    regime_applicability: str
    efiling_mapping: str


def _clean_value(value: Any) -> str:
    if value is None:
        return "Not specified in the source table"
    if isinstance(value, float) and value != value:
        return "Not specified in the source table"
    text = str(value).strip()
    return text if text else "Not specified in the source table"


def normalize_record(row: dict[str, Any]) -> TaxFormRecord:
    """Normalize source workbook fields without inventing missing values."""

    return TaxFormRecord(
        form_no=_clean_value(row.get("Form No")),
        category=_clean_value(row.get("Category")),
        purpose=_clean_value(row.get("Purpose")),
        applicable_sections=_clean_value(row.get("Applicable Sections")),
        filing_frequency=_clean_value(row.get("Filing Frequency")),
        due_dates=_clean_value(row.get("Due Dates")),
        user_categories=_clean_value(row.get("User Categories")),
        regime_applicability=_clean_value(row.get("Old/New Regime Applicability")),
        efiling_mapping=_clean_value(row.get("API / E-Filing Mapping Possibilities")),
    )


def _chat_example(
    *,
    record: TaxFormRecord,
    question: str,
    answer: str,
    qa_type: str,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "form_no": record.form_no,
        "category": record.category,
        "qa_type": qa_type,
        "source_fields": {
            "purpose": record.purpose,
            "applicable_sections": record.applicable_sections,
            "filing_frequency": record.filing_frequency,
            "due_dates": record.due_dates,
            "user_categories": record.user_categories,
            "regime_applicability": record.regime_applicability,
            "efiling_mapping": record.efiling_mapping,
        },
    }


def examples_for_record(record: TaxFormRecord) -> list[dict[str, Any]]:
    """Create high-signal QA examples for one tax-form reference row."""

    form = record.form_no
    return [
        _chat_example(
            record=record,
            qa_type="purpose",
            question=f"What is {form} used for?",
            answer=f"{form} is used for {record.purpose}.",
        ),
        _chat_example(
            record=record,
            qa_type="applicability",
            question=f"Who should use {form}, and which section applies?",
            answer=(
                f"{form} applies to {record.user_categories}. "
                f"The applicable section is {record.applicable_sections}."
            ),
        ),
        _chat_example(
            record=record,
            qa_type="filing_timeline",
            question=f"How often is {form} filed and what is its due date?",
            answer=(
                f"{form} has {record.filing_frequency} filing frequency. "
                f"The due date is {record.due_dates}."
            ),
        ),
        _chat_example(
            record=record,
            qa_type="regime",
            question=f"Is {form} applicable under the old regime, new regime, or both?",
            answer=(
                f"For {form}, old/new regime applicability is "
                f"{record.regime_applicability}."
            ),
        ),
        _chat_example(
            record=record,
            qa_type="efiling_mapping",
            question=f"What e-filing or API mapping is listed for {form}?",
            answer=f"The listed e-filing/API mapping for {form} is {record.efiling_mapping}.",
        ),
        _chat_example(
            record=record,
            qa_type="summary",
            question=f"Summarize the Income Tax Master entry for {form}.",
            answer=(
                f"{form} is a {record.category} form used for {record.purpose}. "
                f"It applies to {record.user_categories}, references "
                f"{record.applicable_sections}, is filed {record.filing_frequency}, "
                f"has due date {record.due_dates}, regime applicability "
                f"{record.regime_applicability}, and maps to {record.efiling_mapping}."
            ),
        ),
    ]


def build_income_tax_qa_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build chat-style SFT examples from workbook rows."""

    examples: list[dict[str, Any]] = []
    for row in rows:
        examples.extend(examples_for_record(normalize_record(row)))
    return examples
