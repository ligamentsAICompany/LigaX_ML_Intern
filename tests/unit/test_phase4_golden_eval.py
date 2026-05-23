"""Phase 4 Golden Eval Generation tests."""

from __future__ import annotations

import json

from agent.core.dataset_inspection import build_dataset_profile
from agent.core.golden_eval import NOT_SPECIFIED, generate_golden_eval_cases


def _case_by_question(cases: list[dict], question_part: str) -> dict:
    return next(case for case in cases if question_part in case["question"])


def test_income_tax_reference_table_generates_source_grounded_eval_cases():
    rows = [
        {
            "Form No": "ITR-1",
            "Category": "Income Tax Return",
            "Purpose": "Simple income return",
            "Due Dates": "31 July",
            "Old/New Regime Applicability": "Both",
        },
        {
            "Form No": "Form 16",
            "Category": "TDS certificate",
            "Purpose": "Salary TDS certificate",
            "Due Dates": "15 June",
            "Old/New Regime Applicability": "Both",
        },
        {
            "Form No": "Form 10E",
            "Category": "Relief",
            "Purpose": "Salary arrear relief",
            "Due Dates": "",
            "Old/New Regime Applicability": None,
        },
        {
            "Form No": "Form 49B",
            "Category": "TAN",
            "Purpose": "TAN application",
            "Due Dates": "",
            "Old/New Regime Applicability": "",
        },
    ]

    cases = [
        case.to_dict()
        for case in generate_golden_eval_cases(
            rows=rows,
            columns=list(rows[0]),
            dataset_shape="structured_reference_table",
            source={"type": "local_file", "path": "Income_Tax_Master.xlsx"},
            domain="income_tax",
        )
    ]

    assert (
        _case_by_question(cases, "What is ITR-1 used for?")["expected_answer"]
        == "Simple income return"
    )
    assert (
        _case_by_question(cases, "What is Form 16 used for?")["expected_answer"]
        == "Salary TDS certificate"
    )
    assert (
        _case_by_question(cases, "What is Form 10E used for?")["expected_answer"]
        == "Salary arrear relief"
    )
    assert (
        _case_by_question(cases, "Which form is used for TAN application?")[
            "expected_answer"
        ]
        == "Form 49B"
    )
    assert (
        _case_by_question(cases, "due date is listed for Form 10E?")["expected_answer"]
        == NOT_SPECIFIED
    )
    assert (
        _case_by_question(cases, "regime applicability is listed for Form 10E?")[
            "expected_answer"
        ]
        == NOT_SPECIFIED
    )

    itr_case = _case_by_question(cases, "What is ITR-1 used for?")
    assert itr_case["source_metadata"]["row_index"] == 0
    assert itr_case["source_metadata"]["columns_used"] == ["Form No", "Purpose"]
    assert itr_case["source_metadata"]["source_fields"] == {
        "Form No": "ITR-1",
        "Purpose": "Simple income return",
    }
    assert itr_case["domain"] == "income_tax"
    assert itr_case["task_type"] == "structured_reference_qa"
    assert "supported by the source fields" in " ".join(itr_case["rubric"]).lower()
    assert "Do not hallucinate values" in itr_case["quality_constraints"]


def test_prompt_completion_rows_normalize_to_eval_cases():
    rows = [
        {
            "prompt": "Explain section 80C.",
            "completion": "Section 80C covers eligible tax deductions.",
        }
    ]

    cases = generate_golden_eval_cases(
        rows=rows,
        columns=["prompt", "completion"],
        dataset_shape="prompt_completion",
    )

    assert len(cases) == 1
    case = cases[0].to_dict()
    assert case["question"] == "Explain section 80C."
    assert case["expected_answer"] == "Section 80C covers eligible tax deductions."
    assert case["task_type"] == "sft_reference_eval"
    assert case["source_metadata"]["holdout_status"] == "sample_based/unverified"
    assert case["source_metadata"]["columns_used"] == ["prompt", "completion"]


def test_messages_rows_normalize_to_eval_cases():
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Answer as a tax assistant."},
                {"role": "user", "content": "What is Form 16?"},
                {"role": "assistant", "content": "A salary TDS certificate."},
            ]
        }
    ]

    cases = generate_golden_eval_cases(
        rows=rows,
        columns=["messages"],
        dataset_shape="sft_messages",
    )

    assert len(cases) == 1
    case = cases[0].to_dict()
    assert case["question"] == "What is Form 16?"
    assert case["expected_answer"] == "A salary TDS certificate."
    assert case["source_metadata"]["columns_used"] == ["messages"]
    assert case["source_metadata"]["source_fields"]["messages"][0]["role"] == "system"


def test_sft_rows_only_use_holdout_label_for_explicit_eval_sources():
    rows = [
        {
            "prompt": "Explain section 80D.",
            "completion": "Section 80D covers medical insurance deductions.",
        }
    ]

    cases = generate_golden_eval_cases(
        rows=rows,
        columns=["prompt", "completion"],
        dataset_shape="prompt_completion",
        source={"split": "validation"},
    )

    assert len(cases) == 1
    case = cases[0].to_dict()
    assert case["task_type"] == "sft_holdout"
    assert case["source_metadata"]["holdout_status"] == "explicit_holdout"


def test_multi_turn_messages_preserve_context_before_answered_user_turn():
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Answer as a tax assistant."},
                {"role": "user", "content": "We are discussing Indian tax forms."},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": "Which form reports salary TDS?"},
                {"role": "assistant", "content": "Form 16 reports salary TDS."},
            ]
        }
    ]

    cases = generate_golden_eval_cases(
        rows=rows,
        columns=["messages"],
        dataset_shape="sft_messages",
    )

    assert len(cases) == 1
    case = cases[0].to_dict()
    assert case["question"] == "Which form reports salary TDS?"
    assert case["expected_answer"] == "Form 16 reports salary TDS."
    assert case["source_metadata"]["source_fields"]["messages"] == [
        {"role": "system", "content": "Answer as a tax assistant."},
        {"role": "user", "content": "We are discussing Indian tax forms."},
        {"role": "user", "content": "Which form reports salary TDS?"},
        {"role": "assistant", "content": "Form 16 reports salary TDS."},
    ]


def test_malformed_message_rows_are_skipped():
    cases = generate_golden_eval_cases(
        rows=[
            {"messages": "{not valid json"},
            {"messages": [{"role": "user", "content": "Missing assistant."}]},
            {"messages": [{"role": "assistant", "content": "Missing user."}]},
        ],
        columns=["messages"],
        dataset_shape="sft_messages",
    )

    assert cases == []


def test_structured_templates_skip_rows_missing_required_keys():
    rows = [
        {"Form No": "", "Purpose": "Salary TDS certificate", "Due Dates": "15 June"},
        {"Form No": "ITR-1", "Purpose": "", "Due Dates": "31 July"},
        {"Form No": "Form 16", "Purpose": "Salary TDS certificate", "Due Dates": ""},
    ]

    cases = [
        case.to_dict()
        for case in generate_golden_eval_cases(
            rows=rows,
            columns=["Form No", "Purpose", "Due Dates"],
            dataset_shape="structured_reference_table",
        )
    ]

    questions = [case["question"] for case in cases]
    assert all(NOT_SPECIFIED not in question for question in questions)
    assert "What is Form 16 used for?" in questions
    assert "Which form is used for Salary TDS certificate?" in questions
    assert "What due date is listed for ITR-1?" in questions
    assert "What due date is listed for Form 16?" in questions
    assert "What is ITR-1 used for?" not in questions
    assert len(cases) == 4


def test_max_cases_limit_is_applied_after_skipping_invalid_cases():
    rows = [
        {"Form No": "", "Purpose": "Missing form"},
        {"Form No": "ITR-1", "Purpose": "Simple income return"},
        {"Form No": "Form 16", "Purpose": "Salary TDS certificate"},
    ]

    cases = generate_golden_eval_cases(
        rows=rows,
        columns=["Form No", "Purpose"],
        dataset_shape="structured_reference_table",
        max_cases=2,
    )

    assert len(cases) == 2
    assert [case.question for case in cases] == [
        "What is ITR-1 used for?",
        "Which form is used for Simple income return?",
    ]


def test_dataset_profile_exposes_redacted_golden_eval_cases():
    rows = [
        {
            "prompt": "Use token hf_abcdefghijklmnopqrstuvwxyz1234567890 safely.",
            "completion": "Contact me at user@example.com.",
        }
    ]

    profile = build_dataset_profile(
        rows=rows,
        source={"type": "local_file", "path": "sft.jsonl", "format": "jsonl"},
    )

    payload = json.dumps(profile["golden_eval"])
    assert "hf_abcdefghijklmnopqrstuvwxyz1234567890" not in payload
    assert "user@example.com" not in payload
    assert "[REDACTED_" in payload
    assert profile["golden_eval"]["case_count"] == 1
    assert profile["golden_eval"]["cases"][0]["question"].startswith("Use token")
