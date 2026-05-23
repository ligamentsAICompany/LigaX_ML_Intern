"""Tests for the one-run automatic HF Jobs fine-tuning flow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.config import Config
from agent.core import cost_estimation
from agent.core.auto_finetune import (
    AUTO_FINETUNE_COST_CAP_USD,
    AutoFineTuneBlock,
    AutoFineTunePipeline,
    apply_known_training_repair,
    build_auto_finetune_job_args,
    classify_known_training_error,
    credential_readiness,
    is_auto_finetune_intent,
    repair_known_training_error,
    resolve_auto_finetune_config,
    resolve_dataset_repo,
)
from agent.core.script_smoke import run_script_smoke
from agent.core.training_templates import render_sft_training_script


def test_auto_finetune_config_caps_spend_at_five_dollars():
    config = Config(model_name="test-model", auto_finetune_cost_cap_usd=50)

    resolved = resolve_auto_finetune_config(config)

    assert resolved.provider_id == "hf-jobs"
    assert resolved.cost_cap_usd == AUTO_FINETUNE_COST_CAP_USD
    assert resolved.max_retries == 1


def test_credential_readiness_exposes_only_booleans():
    readiness = credential_readiness(
        {"HF_TOKEN": "hf_secret_value", "OPENAI_API_KEY": "sk-secret"}
    )

    assert readiness == {
        "hf_token_configured": True,
        "openai_api_key_configured": True,
    }
    assert "hf_secret_value" not in str(readiness)
    assert "sk-secret" not in str(readiness)


def test_intent_and_dataset_repo_resolution_from_text_and_context():
    text = (
        "Please fine tune this dataset https://huggingface.co/datasets/acme/support-qa"
    )

    assert is_auto_finetune_intent(text)
    assert resolve_dataset_repo(text, {}) == "acme/support-qa"
    assert (
        resolve_dataset_repo("fine tune this dataset", {"dataset_repo": "org/data"})
        == "org/data"
    )


def test_training_template_uses_current_compatible_apis_and_passes_smoke():
    script = render_sft_training_script(
        dataset_repo="acme/support-qa",
        output_model_repo="ligaments-dev/support-qa-auto-sft",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        max_length=1024,
        trackio_project="ml-intern-auto-finetune",
        trackio_run_name="support-qa-auto-sft",
    )

    assert "max_length=1024" in script
    assert "max_seq_length" not in script
    assert "eval_strategy=" in script
    assert "evaluation_strategy" not in script
    assert "processing_class=tokenizer" in script
    assert "tokenizer=tokenizer" not in script
    assert "import trackio" not in script
    assert "trackio.init" not in script
    assert "report_to=[]" in script
    assert '"trackio"' not in script
    assert "project_name" not in script
    assert "run_name" not in script

    smoke = run_script_smoke(
        script,
        job_args={
            "operation": "run",
            "dependencies": ["transformers", "trl", "torch", "datasets"],
            "timeout": "3h",
        },
    )
    assert smoke.passed, smoke.to_dict()
    assert smoke.status == "pass"


def test_trackio_rank_pattern_failure_is_classified_and_repaired():
    script = """
import trackio
from trl import SFTConfig, SFTTrainer

trackio.init(project="auto")
config = SFTConfig(
    output_dir="out",
    push_to_hub=True,
    hub_model_id="user/model",
    report_to=["trackio"],
)
trainer = SFTTrainer(args=config)
trainer.train()
""".strip()
    error = """
pyarrow.lib.ArrowNotImplementedError: Cannot write struct type 'rank_pattern' with no child field to Parquet.
transformers/integrations/integration_utils.py on_push_begin -> self._trackio.sync
""".strip()

    assert classify_known_training_error(error) == "trackio_sync_failure"
    repaired = repair_known_training_error(script, error)

    assert repaired is not None
    assert "trackio.init" not in repaired
    assert "import trackio" not in repaired
    assert "report_to=[]" in repaired


def test_trackio_rank_pattern_repair_removes_job_trackio_metadata():
    args = build_auto_finetune_job_args(
        dataset_repo="org/data",
        output_model_repo="ligaments-dev/data-auto-sft",
        config=resolve_auto_finetune_config(Config(model_name="test-model")),
    )
    args["script"] = args["script"].replace("report_to=[]", 'report_to=["trackio"]')
    args["dependencies"] = [*args["dependencies"], "trackio"]
    args["trackio"] = {"project": "auto", "name": "run"}

    repaired = apply_known_training_repair(
        args,
        "Cannot write struct type 'rank_pattern' with no child field to Parquet; trackio sync failed",
    )

    assert repaired is not None
    assert "trackio" not in repaired["dependencies"]
    assert "trackio" not in repaired
    assert "report_to=[]" in repaired["script"]


@pytest.mark.asyncio
async def test_pipeline_blocks_unknown_cost_without_approval(monkeypatch):
    async def fake_estimate(*_args, **_kwargs):
        return cost_estimation.CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason="No price is available.",
            label="mystery-gpu",
        )

    monkeypatch.setattr("agent.core.auto_finetune.estimate_tool_cost", fake_estimate)
    pipeline = AutoFineTunePipeline(config=Config(model_name="test-model"))

    result = await pipeline.prepare(
        "fine tune this dataset", {"dataset_repo": "org/data"}
    )

    assert isinstance(result, AutoFineTuneBlock)
    assert result.error_code == "auto_finetune_cost_unknown"
    assert result.approval_required is False
    assert "No price" in result.message


@pytest.mark.asyncio
async def test_pipeline_blocks_document_corpus_strategy_before_sft(monkeypatch):
    async def fake_estimate(*_args, **_kwargs):  # pragma: no cover - guard
        raise AssertionError(
            "cost estimation should not run for blocked document corpus"
        )

    monkeypatch.setattr("agent.core.auto_finetune.estimate_tool_cost", fake_estimate)
    pipeline = AutoFineTunePipeline(config=Config(model_name="test-model"))

    result = await pipeline.prepare(
        "fine tune this dataset",
        {
            "dataset_repo": "org/pdf-corpus",
            "dataset_profile": {
                "inferred_shape": "document_corpus",
                "strategy": {
                    "strategy": "rag",
                    "can_train_without_override": False,
                    "requires_user_override_for_training": True,
                    "override_message": "Use retrieval for uploaded documents.",
                },
            },
        },
    )

    assert isinstance(result, AutoFineTuneBlock)
    assert result.error_code == "auto_finetune_strategy_blocked"
    assert "retrieval" in result.message.lower()


@pytest.mark.asyncio
async def test_pipeline_blocks_total_retry_cost_over_cap(monkeypatch):
    async def fake_estimate(*_args, **_kwargs):
        return cost_estimation.CostEstimate(
            estimated_cost_usd=3.0,
            billable=True,
            label="t4-small",
        )

    monkeypatch.setattr("agent.core.auto_finetune.estimate_tool_cost", fake_estimate)
    pipeline = AutoFineTunePipeline(config=Config(model_name="test-model"))

    result = await pipeline.prepare(
        "fine tune this dataset", {"dataset_repo": "org/data"}
    )

    assert isinstance(result, AutoFineTuneBlock)
    assert result.error_code == "auto_finetune_cost_cap_exceeded"
    assert result.approval_required is False
    assert "$5.00" in result.message


@pytest.mark.asyncio
async def test_pipeline_prepares_hf_jobs_only_arguments_under_cap(monkeypatch):
    async def fake_estimate(*_args, **_kwargs):
        return cost_estimation.CostEstimate(
            estimated_cost_usd=1.5,
            billable=True,
            label="t4-small",
        )

    monkeypatch.setattr("agent.core.auto_finetune.estimate_tool_cost", fake_estimate)
    pipeline = AutoFineTunePipeline(config=Config(model_name="test-model"))

    plan = await pipeline.prepare(
        "fine tune this dataset https://huggingface.co/datasets/org/data",
        {},
    )

    assert not isinstance(plan, AutoFineTuneBlock)
    assert plan.provider_id == "hf-jobs"
    assert plan.estimated_total_cost_usd == 3.0
    assert plan.job_args["operation"] == "run"
    assert plan.job_args["provider_id"] == "hf-jobs"
    assert plan.job_args["hardware_flavor"] == "t4-small"
    assert plan.job_args["script_smoke"]["passed"] is True
    assert plan.approval_required is False


def test_build_auto_finetune_job_args_never_includes_secret_values():
    args = build_auto_finetune_job_args(
        dataset_repo="org/data",
        output_model_repo="ligaments-dev/data-auto-sft",
        config=resolve_auto_finetune_config(Config(model_name="test-model")),
    )

    assert args["provider_id"] == "hf-jobs"
    assert args["secrets"] == {"HF_TOKEN": "$HF_TOKEN"}
    assert "OPENAI_API_KEY" not in str(args)


@pytest.mark.asyncio
async def test_run_emits_progress_and_final_links_without_approval(monkeypatch):
    async def fake_estimate(*_args, **_kwargs):
        return cost_estimation.CostEstimate(
            estimated_cost_usd=1.0,
            billable=True,
            label="t4-small",
        )

    async def fake_executor(_args):
        return SimpleNamespace(
            ok=True,
            output=(
                "Python job completed!\n"
                "**Job ID:** abc123\n"
                "**Final Status:** COMPLETED\n"
                "**View at:** https://huggingface.co/jobs/user/abc123\n"
            ),
            job_url="https://huggingface.co/jobs/user/abc123",
            final_status="COMPLETED",
        )

    class FakeSession:
        def __init__(self):
            self.events = []
            self.hf_token = "hf_secret"

        async def send_event(self, event):
            self.events.append(event)

    monkeypatch.setattr("agent.core.auto_finetune.estimate_tool_cost", fake_estimate)
    session = FakeSession()
    pipeline = AutoFineTunePipeline(
        config=Config(model_name="test-model"),
        executor=fake_executor,
        env={"HF_TOKEN": "hf_secret"},
    )

    result = await pipeline.run(
        session,
        "fine tune this dataset",
        {"dataset_repo": "org/data"},
    )

    assert result.success is True
    assert result.approval_required is False
    event_types = [event.event_type for event in session.events]
    assert "approval_required" not in event_types
    assert event_types.count("auto_finetune_progress") >= 3
    assert "assistant_message" in event_types
    assert result.model_repo_url.startswith("https://huggingface.co/")
    assert "hf_secret" not in str([event.data for event in session.events])
