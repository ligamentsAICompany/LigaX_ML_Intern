"""Phase 3 Strategy Selector tests."""

from __future__ import annotations

import json

from agent.core.dataset_inspection import build_dataset_profile, inspect_local_dataset
from agent.core.strategy_selector import select_ml_strategy
from agent.tools.dataset_tools import (
    _build_hub_dataset_profile,
    _format_strategy_selector,
)


def _messages_row(
    user: str = "What is section 80C?", assistant: str = "A deduction."
) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Answer as a tax assistant."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def test_income_tax_reference_table_selects_rag_not_blind_fine_tune():
    profile = build_dataset_profile(
        rows=[
            {
                "Section": "80C",
                "Description": "Deduction for eligible investments",
                "Limit": "150000",
                "Applicability": "Individuals and HUF",
            },
            {
                "Section": "80D",
                "Description": "Medical insurance deduction",
                "Limit": "25000",
                "Applicability": "Individuals",
            },
        ],
        source={
            "type": "local_file",
            "path": "Income_Tax_Master.xlsx",
            "format": "xlsx",
        },
    )

    decision = select_ml_strategy(profile)

    assert decision.strategy == "rag"
    assert decision.can_train_without_override is False
    assert decision.requires_user_override_for_training is True
    assert any("reference table" in reason.lower() for reason in decision.reasons)
    assert any(
        "retrieval" in action.lower() for action in decision.required_next_actions
    )


def test_valid_large_sft_dataset_selects_fine_tune_with_low_risk():
    profile = build_dataset_profile(
        rows=[_messages_row(), _messages_row("What is 80D?", "Medical insurance.")]
        * 600,
        source={"type": "local_file", "path": "sft.jsonl", "format": "jsonl"},
    )

    decision = select_ml_strategy(profile)

    assert decision.strategy == "fine_tune"
    assert decision.method_hint == "sft"
    assert decision.risk_level == "low"
    assert decision.confidence >= 0.85
    assert decision.can_train_without_override is True


def test_preference_dataset_selects_fine_tune_with_dpo_hint():
    profile = build_dataset_profile(
        rows=[
            {
                "prompt": "Draft a concise tax answer.",
                "chosen": "Section 80C allows deductions up to the statutory limit.",
                "rejected": "Taxes are confusing, ask someone else.",
            }
        ]
        * 800,
        source={"type": "local_file", "path": "dpo.jsonl", "format": "jsonl"},
    )

    decision = select_ml_strategy(profile)

    assert decision.strategy == "fine_tune"
    assert decision.method_hint == "dpo"
    assert decision.metadata["method_recommendation"] == "dpo"
    assert decision.can_train_without_override is True


def test_prompt_only_dataset_selects_fine_tune_with_grpo_hint():
    profile = build_dataset_profile(
        rows=[{"prompt": "Compute deduction eligibility for scenario A."}] * 1200,
        source={"type": "local_file", "path": "grpo.jsonl", "format": "jsonl"},
    )

    decision = select_ml_strategy(profile)

    assert decision.strategy == "fine_tune"
    assert decision.method_hint == "grpo"
    assert decision.metadata["method_recommendation"] == "grpo"
    assert decision.can_train_without_override is False
    assert decision.requires_user_override_for_training is True
    assert any("reward" in action.lower() for action in decision.required_next_actions)
    assert "reward" in decision.override_message.lower()


def test_malformed_dpo_samples_do_not_train_from_column_hint():
    profile = build_dataset_profile(
        rows=[
            {
                "prompt": "Draft a concise tax answer.",
                "chosen": "",
                "rejected": "Taxes are confusing, ask someone else.",
            }
        ]
        * 800,
        source={"type": "local_file", "path": "malformed_dpo.jsonl", "format": "jsonl"},
    )

    decision = select_ml_strategy(profile)

    assert decision.strategy == "data_needed"
    assert decision.can_train_without_override is False
    assert decision.requires_user_override_for_training is True
    assert any("sample" in reason.lower() for reason in decision.reasons)


def test_hub_schema_only_profile_is_not_overconfident_from_columns():
    profile = _build_hub_dataset_profile(
        dataset="org/schema-only-dpo",
        config="default",
        split="train",
        splits_data={
            "splits": [
                {
                    "config": "default",
                    "split": "train",
                    "num_examples": 1200,
                }
            ]
        },
        info_data={
            "dataset_info": {
                "features": {
                    "prompt": {"dtype": "string"},
                    "chosen": {"dtype": "string"},
                    "rejected": {"dtype": "string"},
                }
            }
        },
        rows_data={"rows": []},
        file_format="parquet",
    )

    decision = select_ml_strategy(profile)

    assert decision.strategy == "data_needed"
    assert decision.can_train_without_override is False
    assert decision.requires_user_override_for_training is True
    assert any("sample" in reason.lower() for reason in decision.reasons)


def test_dirty_empty_dataset_selects_data_needed():
    profile = build_dataset_profile(
        rows=[
            {"prompt": "", "completion": ""},
            {"prompt": "", "completion": ""},
        ],
        source={"type": "local_file", "path": "dirty.csv", "format": "csv"},
    )

    decision = select_ml_strategy(profile)

    assert decision.strategy == "data_needed"
    assert decision.risk_level == "high"
    assert decision.can_train_without_override is False
    assert decision.requires_user_override_for_training is True
    assert any(
        "quality" in action.lower() or "collect" in action.lower()
        for action in decision.required_next_actions
    )


def test_tiny_high_risk_sft_exposes_override_message():
    profile = build_dataset_profile(
        rows=[_messages_row(), _messages_row("What is 80D?", "Medical insurance.")],
        source={"type": "local_file", "path": "tiny_sft.jsonl", "format": "jsonl"},
    )

    decision = select_ml_strategy(profile)

    assert decision.strategy == "fine_tune"
    assert decision.risk_level == "high"
    assert decision.can_train_without_override is False
    assert decision.requires_user_override_for_training is True
    assert "override" in decision.override_message.lower()
    assert "high-risk fine-tune" in decision.override_message.lower()


def test_dataset_profile_exposes_strategy_decision(tmp_path):
    path = tmp_path / "sft.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in [_messages_row(), _messages_row()] * 600),
        encoding="utf-8",
    )

    profile = inspect_local_dataset(path)

    assert profile["strategy"]["strategy"] == "fine_tune"
    assert profile["strategy"]["method_hint"] == "sft"
    assert profile["strategy"]["can_train_without_override"] is True


def test_dataset_tool_formats_strategy_selector_result():
    formatted = _format_strategy_selector(
        {
            "row_count": 2,
            "columns": ["Section", "Description", "Limit", "Applicability"],
            "format": "xlsx",
            "sample_rows": [
                {
                    "Section": "80C",
                    "Description": "Deduction",
                    "Limit": "150000",
                    "Applicability": "Individuals",
                }
            ],
        }
    )

    assert "## Strategy Selector" in formatted
    assert "Strategy: rag" in formatted
    assert "Training can proceed without override: no" in formatted
