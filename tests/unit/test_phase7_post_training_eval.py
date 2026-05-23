"""Phase 7 post-training evaluation gate tests."""

from __future__ import annotations

import json

from agent.core.golden_eval import generate_golden_eval_cases
from agent.core.post_training_eval import evaluate_post_training_outputs
from agent.core.provenance import build_artifact_card, build_training_provenance


def _income_tax_cases() -> list[dict]:
    rows = [
        {
            "Form No": "ITR-1",
            "Category": "Income Tax Return",
            "Purpose": "Simple income return",
            "Due Dates": "31 July",
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
    return [
        case.to_dict()
        for case in generate_golden_eval_cases(
            rows=rows,
            columns=list(rows[0]),
            dataset_shape="structured_reference_table",
            source={"type": "local_file", "path": "Income_Tax_Master.xlsx"},
            domain="income_tax",
        )
    ]


def _case_by_question(cases: list[dict], question_part: str) -> dict:
    return next(case for case in cases if question_part in case["question"])


def test_correct_income_tax_answers_pass_post_training_gate():
    cases = _income_tax_cases()
    itr_case = _case_by_question(cases, "What is ITR-1 used for?")
    form_10e_case = _case_by_question(cases, "What is Form 10E used for?")
    tan_case = _case_by_question(cases, "Which form is used for TAN application?")

    report = evaluate_post_training_outputs(
        cases=[itr_case, form_10e_case, tan_case],
        outputs={
            itr_case["id"]: "ITR-1 is used for a simple income return.",
            form_10e_case["id"]: "Form 10E is used for salary arrear relief.",
            tan_case["id"]: "Form 49B is used for TAN application.",
        },
    )

    assert report["status"] == "passed"
    assert report["summary"]["passed"] == 3
    assert all(case["status"] == "passed" for case in report["cases"])


def test_chemistry_answer_for_itr1_fails_as_off_topic_hallucination():
    cases = _income_tax_cases()
    itr_case = _case_by_question(cases, "What is ITR-1 used for?")

    report = evaluate_post_training_outputs(
        cases=[itr_case],
        outputs={
            itr_case["id"]: (
                "ITR-1 is a chemistry worksheet about acids, bases, catalysts, "
                "and laboratory reactions."
            )
        },
    )

    assert report["status"] == "failed"
    result = report["cases"][0]
    assert result["status"] == "failed"
    assert result["score"] == 0.0
    assert "off-topic" in " ".join(result["reasons"]).lower()


def test_w2_answer_for_form_10e_fails_as_factual_mismatch():
    cases = _income_tax_cases()
    form_10e_case = _case_by_question(cases, "What is Form 10E used for?")

    report = evaluate_post_training_outputs(
        cases=[form_10e_case],
        outputs={
            form_10e_case["id"]: (
                "Form 10E is a U.S. W-2 wage statement used by employers."
            )
        },
    )

    assert report["status"] == "failed"
    result = report["cases"][0]
    assert result["status"] == "failed"
    assert "missing expected facts" in " ".join(result["reasons"]).lower()
    assert "w-2" in " ".join(result["reasons"]).lower()


def test_tan_answer_missing_form_49b_fails():
    cases = _income_tax_cases()
    tan_case = _case_by_question(cases, "Which form is used for TAN application?")

    report = evaluate_post_training_outputs(
        cases=[tan_case],
        outputs={
            tan_case["id"]: "A TAN application is submitted on the e-filing portal."
        },
    )

    assert report["status"] == "failed"
    result = report["cases"][0]
    assert result["status"] == "failed"
    assert "form 49b" in " ".join(result["missing_facts"]).lower()


def test_expected_fact_with_unsupported_extra_claim_fails():
    cases = _income_tax_cases()
    itr_case = _case_by_question(cases, "What is ITR-1 used for?")

    report = evaluate_post_training_outputs(
        cases=[itr_case],
        outputs={
            itr_case["id"]: (
                "ITR-1 is used for a simple income return. "
                "It also guarantees an automatic tax refund."
            )
        },
    )

    result = report["cases"][0]
    assert report["status"] == "failed"
    assert result["status"] == "failed"
    assert "unsupported claims" in " ".join(result["reasons"]).lower()


def test_short_expected_fact_paraphrase_passes_when_token_overlap_is_high():
    cases = _income_tax_cases()
    form_10e_case = _case_by_question(cases, "What is Form 10E used for?")

    report = evaluate_post_training_outputs(
        cases=[form_10e_case],
        outputs={
            form_10e_case["id"]: (
                "Form 10E provides relief for salary arrears based on the source."
            )
        },
    )

    assert report["status"] == "passed"
    assert report["cases"][0]["status"] == "passed"


def test_correct_answer_based_on_provided_context_does_not_need_rag():
    cases = _income_tax_cases()
    itr_case = _case_by_question(cases, "What is ITR-1 used for?")

    report = evaluate_post_training_outputs(
        cases=[itr_case],
        outputs={
            itr_case["id"]: (
                "Based on the provided context, ITR-1 is used for a simple income return."
            )
        },
    )

    assert report["status"] == "passed"
    assert report["cases"][0]["status"] == "passed"


def test_mixed_case_report_marks_failed_and_needs_rag_cases():
    cases = _income_tax_cases()
    itr_case = _case_by_question(cases, "What is ITR-1 used for?")
    form_10e_case = _case_by_question(cases, "What is Form 10E used for?")
    tan_case = _case_by_question(cases, "Which form is used for TAN application?")

    report = evaluate_post_training_outputs(
        cases=[itr_case, form_10e_case, tan_case],
        outputs={
            itr_case["id"]: "ITR-1 is used for a simple income return.",
            form_10e_case["id"]: "I need retrieval context before answering Form 10E.",
            tan_case["id"]: "A TAN application is submitted on the e-filing portal.",
        },
    )

    statuses = {case["case_id"]: case["status"] for case in report["cases"]}
    assert report["status"] == "failed"
    assert statuses[itr_case["id"]] == "passed"
    assert statuses[form_10e_case["id"]] == "needs_rag"
    assert statuses[tan_case["id"]] == "failed"


def test_empty_outputs_need_more_data():
    cases = _income_tax_cases()
    report = evaluate_post_training_outputs(cases=cases[:2], outputs={})

    assert report["status"] == "needs_more_data"
    assert report["summary"]["needs_more_data"] == 2
    assert all(case["status"] == "needs_more_data" for case in report["cases"])


def test_report_redacts_token_like_outputs():
    cases = _income_tax_cases()
    itr_case = _case_by_question(cases, "What is ITR-1 used for?")
    token = "hf_abcdefghijklmnopqrstuvwxyz1234567890"

    report = evaluate_post_training_outputs(
        cases=[itr_case],
        outputs={itr_case["id"]: f"Leaked token {token} while answering chemistry."},
    )
    serialized = json.dumps(report)

    assert token not in serialized
    assert "[REDACTED_HF_TOKEN]" in serialized


def test_report_redacts_secret_keys_in_non_string_outputs():
    cases = _income_tax_cases()
    itr_case = _case_by_question(cases, "What is ITR-1 used for?")

    report = evaluate_post_training_outputs(
        cases=[itr_case],
        outputs={
            itr_case["id"]: {
                "answer": "ITR-1 is used for a simple income return.",
                "metadata": {
                    "api_key": "plain-secret-value",
                    "nested": {"access_token": "another-secret-value"},
                },
            }
        },
    )
    serialized = json.dumps(report)

    assert "plain-secret-value" not in serialized
    assert "another-secret-value" not in serialized
    assert serialized.count("[REDACTED_SECRET]") >= 2


def test_provenance_and_artifact_card_include_post_training_eval_summary():
    cases = _income_tax_cases()
    itr_case = _case_by_question(cases, "What is ITR-1 used for?")
    eval_report = evaluate_post_training_outputs(
        cases=[itr_case],
        outputs={itr_case["id"]: "ITR-1 is used for chemistry lab notes."},
    )

    provenance = build_training_provenance(
        base_model="google/gemma-2-2b-it",
        dataset_profile={
            "row_count": 3,
            "columns": ["Form No", "Purpose"],
            "golden_eval": {"cases": cases, "case_count": len(cases)},
        },
        training_method="sft_lora",
        post_training_eval=eval_report,
    )
    card = build_artifact_card(provenance)

    assert provenance["post_training_eval"]["status"] == "failed"
    assert provenance["post_training_eval"]["case_count"] == 1
    assert "Post-Training Eval: failed" in card


def test_inconsistent_provenance_eval_mapping_is_marked_unverified():
    provenance = build_training_provenance(
        base_model="google/gemma-2-2b-it",
        dataset_profile={"row_count": 1, "columns": ["question", "answer"]},
        training_method="sft_lora",
        post_training_eval={
            "status": "passed",
            "case_count": 2,
            "summary": {"passed": 1, "failed": 1},
            "cases": [{"case_id": "case-1", "status": "failed", "reasons": ["bad"]}],
        },
    )
    card = build_artifact_card(provenance)

    assert provenance["post_training_eval"]["status"] == "unverified"
    assert provenance["post_training_eval"]["valid"] is False
    assert "Post-Training Eval: unverified" in card
