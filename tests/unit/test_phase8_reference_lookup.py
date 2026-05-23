"""Phase 8 Reference Lookup / RAG path tests."""

from __future__ import annotations

import json

from agent.core.dataset_inspection import build_dataset_profile
from agent.core.reference_lookup import NOT_SPECIFIED, build_reference_index


INCOME_TAX_ROWS = [
    {
        "Form No": "ITR-1 (Sahaj)",
        "Category": "Income Tax Return",
        "Purpose": "Simple income return",
        "Applicable Sections": "Sec 139(1)",
        "Filing Frequency": "Annual",
        "Due Dates": "31 July",
        "User Categories": "Salaried Individuals",
        "Old/New Regime Applicability": "Both",
        "API / E-Filing Mapping Possibilities": "ITR Filing API",
    },
    {
        "Form No": "Form 16",
        "Category": "TDS certificate",
        "Purpose": "Salary TDS certificate",
        "Applicable Sections": "Sec 203",
        "Filing Frequency": "Annual",
        "Due Dates": "15 June",
        "User Categories": "Salaried Individuals",
        "Old/New Regime Applicability": "Both",
        "API / E-Filing Mapping Possibilities": "TDS certificate download",
    },
    {
        "Form No": "Form 10E",
        "Category": "Relief",
        "Purpose": "Salary arrear relief",
        "Applicable Sections": "Sec 89",
        "Filing Frequency": "As needed",
        "Due Dates": "",
        "User Categories": "Individuals with salary arrears",
        "Old/New Regime Applicability": None,
        "API / E-Filing Mapping Possibilities": "E-Filing portal",
    },
    {
        "Form No": "Form 49B",
        "Category": "TAN",
        "Purpose": "TAN application",
        "Applicable Sections": "Sec 203A",
        "Filing Frequency": "As needed",
        "Due Dates": "",
        "User Categories": "Tax deductors and collectors",
        "Old/New Regime Applicability": "",
        "API / E-Filing Mapping Possibilities": "TAN application portal",
    },
]


def _index():
    return build_reference_index(
        INCOME_TAX_ROWS,
        columns=list(INCOME_TAX_ROWS[0]),
        source={"type": "local_file", "path": "Income_Tax_Master.xlsx"},
    )


def test_income_tax_lookup_answers_exact_and_reverse_questions():
    index = _index()

    assert index.answer("What is ITR-1 used for?").to_dict()["answer"] == (
        "Simple income return"
    )
    assert index.answer("What is Form 16 used for?").to_dict()["answer"] == (
        "Salary TDS certificate"
    )
    assert index.answer("What is Form 10E used for?").to_dict()["answer"] == (
        "Salary arrear relief"
    )
    assert (
        index.answer("Which form is used for TAN application?").to_dict()["answer"]
        == "Form 49B"
    )


def test_missing_fields_return_not_specified_without_hallucinating():
    result = _index().answer("What due date is listed for Form 10E?").to_dict()

    assert result["status"] == "no_answer"
    assert result["answer"] == NOT_SPECIFIED
    assert result["source_metadata"]["row_index"] == 2
    assert result["source_metadata"]["columns_used"] == ["Form No", "Due Dates"]


def test_unsupported_question_does_not_hallucinate():
    result = _index().answer("What penalty applies if I file ITR-1 late?").to_dict()

    assert result["status"] == "unsupported"
    assert result["answer"] == NOT_SPECIFIED
    assert result["source_metadata"] == {}


def test_source_metadata_includes_row_index_columns_and_source_fields():
    result = _index().answer("Who should use Form 16?").to_dict()

    assert result["status"] == "answered"
    assert result["answer"] == "Salaried Individuals"
    assert result["source_metadata"]["row_index"] == 1
    assert result["source_metadata"]["columns_used"] == ["Form No", "User Categories"]
    assert result["source_metadata"]["source_fields"] == {
        "Form No": "Form 16",
        "User Categories": "Salaried Individuals",
    }
    assert result["source_metadata"]["source"]["path"] == "Income_Tax_Master.xlsx"


def test_query_matching_handles_case_spacing_and_form_variants():
    index = _index()

    assert index.answer("what is itr 1 used for").to_dict()["answer"] == (
        "Simple income return"
    )
    assert index.answer("what is form16 used for").to_dict()["answer"] == (
        "Salary TDS certificate"
    )
    assert (
        index.answer("which form is used for tan  application").to_dict()["answer"]
        == "Form 49B"
    )


def test_form_id_matching_does_not_match_longer_unsupported_variants():
    rows = [INCOME_TAX_ROWS[0]]
    index = build_reference_index(rows, columns=list(rows[0]))

    itr_10_result = index.answer("What is ITR-10 used for?").to_dict()
    itr_1a_result = index.answer("What is ITR-1A used for?").to_dict()

    assert itr_10_result["status"] == "unsupported"
    assert itr_10_result["answer"] == NOT_SPECIFIED
    assert itr_10_result["source_metadata"] == {}
    assert itr_1a_result["status"] == "unsupported"
    assert itr_1a_result["answer"] == NOT_SPECIFIED
    assert itr_1a_result["source_metadata"] == {}


def test_conflicting_duplicate_top_matches_return_ambiguous_no_answer():
    rows = [
        {"Form No": "Form DUP", "Purpose": "First sourced answer"},
        {"Form No": "Form DUP", "Purpose": "Second sourced answer"},
    ]
    index = build_reference_index(rows, source={"path": "duplicates.xlsx"})

    result = index.answer("What is Form DUP used for?").to_dict()

    assert result["status"] == "no_answer"
    assert result["answer"] == NOT_SPECIFIED
    assert result["source_metadata"]["reason"] == "ambiguous_top_match"
    assert result["source_metadata"]["row_indices"] == [0, 1]
    assert result["source_metadata"]["answer_column"] == "Purpose"
    assert result["source_metadata"]["source"]["path"] == "duplicates.xlsx"


def test_lookup_redacts_token_like_row_values():
    rows = [
        {
            "Form No": "Form X",
            "Category": "Secret",
            "Purpose": "hf_abcdefghijklmnopqrstuvwxyz1234567890",
            "Applicable Sections": "Sec X",
            "Filing Frequency": "Annual",
            "User Categories": "Admins",
        }
    ]

    result = build_reference_index(rows).answer("What is Form X used for?").to_dict()
    serialized = json.dumps(result)

    assert "hf_abcdefghijklmnopqrstuvwxyz1234567890" not in serialized
    assert "[REDACTED_HF_TOKEN]" in serialized


def test_lookup_redacts_pii_like_row_values_from_source_metadata():
    rows = [
        {
            "Form No": "Form PII",
            "Purpose": "Collect PAN ABCDE1234F and email owner@example.com",
        }
    ]

    result = build_reference_index(rows).answer("What is Form PII used for?").to_dict()
    serialized = json.dumps(result)

    assert "ABCDE1234F" not in serialized
    assert "owner@example.com" not in serialized
    assert "[REDACTED_PAN]" in serialized
    assert "[REDACTED_EMAIL]" in serialized


def test_non_income_tax_reference_table_answers_from_source_row():
    rows = [
        {
            "Category": "Starter",
            "Description": "Entry-level product tier",
            "Applicability": "Small teams",
        },
        {
            "Category": "Enterprise",
            "Description": "Custom product tier",
            "Applicability": "Large organizations",
        },
    ]
    index = build_reference_index(rows, source={"path": "product_tiers.csv"})

    result = index.answer("What is Starter used for?").to_dict()

    assert result["status"] == "answered"
    assert result["answer"] == "Entry-level product tier"
    assert result["source_metadata"]["row_index"] == 0
    assert result["source_metadata"]["source"]["path"] == "product_tiers.csv"


def test_dataset_profile_exposes_reference_lookup_readiness():
    profile = build_dataset_profile(
        rows=INCOME_TAX_ROWS,
        source={
            "type": "local_file",
            "path": "Income_Tax_Master.xlsx",
            "format": "xlsx",
        },
    )

    assert profile["reference_lookup"]["ready"] is True
    assert profile["reference_lookup"]["row_count"] == 4
    assert "Form No" in profile["reference_lookup"]["key_columns"]
    assert "Purpose" in profile["reference_lookup"]["answer_columns"]
    assert profile["strategy"]["strategy"] == "rag"
