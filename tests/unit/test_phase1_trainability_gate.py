"""Phase 1 Trainability Gate tests."""

from types import SimpleNamespace

import pytest

from agent.config import Config
from agent.core.approval_policy import _is_hf_training_operation, decide_tool_approval
from agent.core.trainability import assess_trainability
from agent.tools.dataset_tools import (
    _extract_split_row_count,
    _format_trainability_gate,
)
from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC


def _config(**overrides):
    return Config(model_name="test-model", **overrides)


def _session(**overrides):
    values = {
        "config": _config(yolo_mode=True),
        "sandbox": None,
        "auto_approval_estimated_spend_usd": 0.0,
        "auto_approval_cost_cap_usd": 5.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tiny_structured_reference_table_is_high_risk_rag_candidate():
    result = assess_trainability(
        {
            "dataset_name": "Income_Tax_Master.xlsx",
            "row_count": 26,
            "columns": ["Section", "Description", "Limit", "Applicability"],
            "format": "xlsx",
            "sample_rows": [
                {
                    "Section": "80C",
                    "Description": "Deduction for eligible investments",
                    "Limit": "150000",
                    "Applicability": "Individuals and HUF",
                }
            ],
        }
    )

    assert result.risk_level == "high"
    assert result.recommendation in {"rag", "hybrid"}
    assert result.score < 50
    assert any("tiny" in reason.lower() for reason in result.reasons)
    assert any("structured reference" in reason.lower() for reason in result.reasons)


@pytest.mark.parametrize(
    "profile",
    [
        {
            "row_count": 1200,
            "columns": ["messages"],
            "format": "jsonl",
            "sample_rows": [
                {
                    "messages": [
                        {"role": "system", "content": "Answer as a tax assistant."},
                        {"role": "user", "content": "What is section 80C?"},
                        {
                            "role": "assistant",
                            "content": "Section 80C covers deductions.",
                        },
                    ]
                }
            ],
        },
        {
            "row_count": 800,
            "columns": ["prompt", "completion"],
            "format": "jsonl",
            "sample_rows": [
                {
                    "prompt": "Explain section 80D.",
                    "completion": "Section 80D covers medical insurance deductions.",
                }
            ],
        },
    ],
)
def test_instruction_datasets_are_fine_tune_candidates(profile):
    result = assess_trainability(profile)

    assert result.recommendation == "fine_tune"
    assert result.risk_level in {"low", "medium"}
    assert result.score >= 60
    assert any(
        "instruction" in reason.lower() or "sft" in reason.lower()
        for reason in result.reasons
    )


def test_missing_dirty_or_empty_dataset_needs_more_data():
    result = assess_trainability(
        {
            "row_count": 0,
            "columns": [],
            "format": "csv",
            "missing_fraction": 0.45,
            "duplicate_fraction": 0.35,
        }
    )

    assert result.recommendation == "data_needed"
    assert result.risk_level == "high"
    assert result.score < 40
    assert any(
        "empty" in reason.lower() or "no rows" in reason.lower()
        for reason in result.reasons
    )
    assert any("missing" in reason.lower() for reason in result.reasons)
    assert any("duplicate" in reason.lower() for reason in result.reasons)


@pytest.mark.parametrize(
    "profile",
    [
        {
            "row_count": 1000,
            "columns": ["messages"],
            "format": "jsonl",
            "sample_rows": [{"messages": []}],
        },
        {
            "row_count": 1000,
            "columns": ["messages"],
            "format": "jsonl",
            "sample_rows": [
                {"messages": [{"role": "user", "content": ""}]},
            ],
        },
        {
            "row_count": 1000,
            "columns": ["prompt", "completion"],
            "format": "jsonl",
            "sample_rows": [{"prompt": "", "completion": "   "}],
        },
    ],
)
def test_malformed_or_empty_instruction_samples_are_not_full_confidence(profile):
    result = assess_trainability(profile)

    assert result.recommendation != "fine_tune"
    assert result.score < 100
    assert any(
        "sample" in reason.lower() or "example" in reason.lower()
        for reason in result.reasons
    )


def test_instruction_schema_without_samples_has_lower_confidence():
    result = assess_trainability(
        {
            "row_count": 1000,
            "columns": ["prompt", "completion"],
            "format": "jsonl",
        }
    )

    assert result.score < 100
    assert result.risk_level != "low"
    assert any("sample" in reason.lower() for reason in result.reasons)


def test_dataset_inspection_can_render_trainability_gate_result():
    formatted = _format_trainability_gate(
        {
            "dataset_name": "Income_Tax_Master.xlsx",
            "row_count": 26,
            "columns": ["Section", "Description", "Limit", "Applicability"],
            "format": "xlsx",
            "sample_rows": [
                {
                    "Section": "80C",
                    "Description": "Deduction for eligible investments",
                    "Limit": "150000",
                    "Applicability": "Individuals and HUF",
                }
            ],
        }
    )

    assert "## Trainability Gate" in formatted
    assert "Recommendation: rag" in formatted
    assert "Risk: high" in formatted
    assert "structured reference table" in formatted


def test_dataset_tools_preserves_zero_row_split_counts():
    row_count = _extract_split_row_count(
        {
            "splits": [
                {
                    "config": "default",
                    "split": "train",
                    "num_examples": 0,
                    "num_rows": 17,
                }
            ]
        },
        "default",
        "train",
    )

    assert row_count == 0


def test_hf_jobs_schema_exposes_trainability_guardrail_metadata():
    description = HF_JOBS_TOOL_SPEC["description"]
    parameters = HF_JOBS_TOOL_SPEC["parameters"]["properties"]

    assert "Trainability Gate" in description
    assert "trainability" in parameters
    assert "dataset_profile" in parameters


@pytest.mark.asyncio
async def test_high_risk_direct_fine_tune_job_is_blocked_before_auto_approval(
    monkeypatch,
):
    async def fake_estimate_tool_cost(*_args, **_kwargs):
        from agent.core.cost_estimation import CostEstimate

        return CostEstimate(estimated_cost_usd=0.05, billable=True, label="t4-small")

    monkeypatch.setattr(
        "agent.core.approval_policy.estimate_tool_cost", fake_estimate_tool_cost
    )

    decision = await decide_tool_approval(
        "hf_jobs",
        {
            "operation": "run",
            "script": (
                "from trl import SFTConfig, SFTTrainer\n"
                "config = SFTConfig(output_dir='out', push_to_hub=True, hub_model_id='user/model')\n"
                "trainer = SFTTrainer(args=config)\n"
                "trainer.train()"
            ),
            "hardware_flavor": "t4-small",
            "timeout": "30m",
            "trainability": {
                "risk_level": "high",
                "recommendation": "rag",
                "reasons": ["Tiny structured reference dataset."],
            },
        },
        session=_session(),
    )

    assert decision.approved is False
    assert decision.requires_approval is True
    assert decision.blocked is True
    assert decision.auto_approval_blocked is True
    assert "Trainability Gate" in decision.block_reason
    assert "rag" in decision.block_reason


@pytest.mark.asyncio
async def test_dataset_profile_is_authoritative_over_spoofed_trainability(
    monkeypatch,
):
    async def fake_estimate_tool_cost(*_args, **_kwargs):
        from agent.core.cost_estimation import CostEstimate

        return CostEstimate(estimated_cost_usd=0.05, billable=True, label="t4-small")

    monkeypatch.setattr(
        "agent.core.approval_policy.estimate_tool_cost", fake_estimate_tool_cost
    )

    decision = await decide_tool_approval(
        "hf_jobs",
        {
            "operation": "run",
            "script": "from trl import SFTTrainer\ntrainer.train()",
            "hardware_flavor": "t4-small",
            "timeout": "30m",
            "trainability": {
                "risk_level": "low",
                "recommendation": "fine_tune",
                "reasons": ["Caller-supplied safe result."],
            },
            "dataset_profile": {
                "row_count": 26,
                "columns": ["Section", "Description", "Limit", "Applicability"],
                "format": "xlsx",
                "sample_rows": [
                    {
                        "Section": "80C",
                        "Description": "Deduction for eligible investments",
                        "Limit": "150000",
                        "Applicability": "Individuals and HUF",
                    }
                ],
            },
        },
        session=_session(),
    )

    assert decision.approved is False
    assert decision.blocked is True
    assert "Trainability Gate" in decision.block_reason
    assert "structured reference" in decision.block_reason.lower()


@pytest.mark.asyncio
async def test_safe_dataset_profile_takes_precedence_over_spoofed_high_risk_result(
    monkeypatch,
):
    async def fake_estimate_tool_cost(*_args, **_kwargs):
        from agent.core.cost_estimation import CostEstimate

        return CostEstimate(estimated_cost_usd=0.05, billable=True, label="t4-small")

    monkeypatch.setattr(
        "agent.core.approval_policy.estimate_tool_cost", fake_estimate_tool_cost
    )

    decision = await decide_tool_approval(
        "hf_jobs",
        {
            "operation": "run",
            "script": (
                "from trl import SFTConfig, SFTTrainer\n"
                "config = SFTConfig(output_dir='out', push_to_hub=True, hub_model_id='user/model')\n"
                "trainer = SFTTrainer(args=config)\n"
                "trainer.train()"
            ),
            "hardware_flavor": "t4-small",
            "timeout": "30m",
            "trainability": {
                "risk_level": "high",
                "recommendation": "rag",
                "reasons": ["Spoofed stale result."],
            },
            "dataset_profile": {
                "row_count": 1200,
                "columns": ["messages"],
                "format": "jsonl",
                "sample_rows": [
                    {
                        "messages": [
                            {"role": "user", "content": "What is section 80C?"},
                            {"role": "assistant", "content": "A deduction section."},
                        ]
                    }
                ],
            },
        },
        session=_session(),
    )

    assert decision.approved is True
    assert decision.auto_approved is True
    assert decision.block_reason is None


@pytest.mark.asyncio
async def test_training_job_without_trainability_metadata_requires_manual_approval(
    monkeypatch,
):
    async def fake_estimate_tool_cost(*_args, **_kwargs):
        from agent.core.cost_estimation import CostEstimate

        return CostEstimate(estimated_cost_usd=0.05, billable=True, label="t4-small")

    monkeypatch.setattr(
        "agent.core.approval_policy.estimate_tool_cost", fake_estimate_tool_cost
    )

    decision = await decide_tool_approval(
        "hf_jobs",
        {
            "operation": "run",
            "script": "python train.py",
            "hardware_flavor": "t4-small",
            "timeout": "30m",
        },
        session=_session(),
    )

    assert decision.approved is False
    assert decision.requires_approval is True
    assert decision.blocked is False
    assert decision.auto_approval_blocked is True
    assert "Trainability Gate" in decision.block_reason


def test_hf_training_detection_allows_transformers_inference_jobs():
    assert (
        _is_hf_training_operation(
            {
                "operation": "run",
                "script": "from transformers import pipeline\nprint(pipeline('sentiment-analysis')('hello'))",
                "dependencies": ["transformers", "torch"],
            }
        )
        is False
    )


@pytest.mark.parametrize(
    "tool_args",
    [
        {"operation": "run", "script": "python train.py"},
        {"operation": "run", "command": ["accelerate", "launch", "sft.py"]},
        {"operation": "uv", "script": "from unsloth import FastLanguageModel"},
        {"operation": "submit", "script": "/app/custom_trainer.py"},
    ],
)
def test_hf_training_detection_catches_obvious_train_commands(tool_args):
    assert _is_hf_training_operation(tool_args) is True
