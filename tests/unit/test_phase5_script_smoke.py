"""Phase 5 tests for static training script smoke validation."""

from types import SimpleNamespace

import pytest

from agent.config import Config
from agent.core.approval_policy import decide_tool_approval
from agent.core.script_smoke import coerce_script_smoke_result, run_script_smoke
from agent.tools.jobs_tool import HfJobsTool, _resolve_uv_command


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


VALID_SFT_SCRIPT = """
import trackio
from trl import SFTConfig, SFTTrainer

trackio.init(project="phase-5")

config = SFTConfig(
    output_dir="out",
    push_to_hub=True,
    hub_model_id="user/model",
)
trainer = SFTTrainer(args=config)
trainer.train()
""".strip()


def test_valid_sft_training_script_with_hub_push_and_trackio_passes():
    result = run_script_smoke(
        VALID_SFT_SCRIPT,
        job_args={
            "operation": "run",
            "timeout": "8h",
            "dependencies": ["trl", "trackio"],
        },
    )

    assert result.passed is True
    assert result.status == "pass"
    assert result.is_training_script is True
    assert result.blocking_issues == []
    assert result.to_dict()["status"] == "pass"


def test_python_syntax_error_fails_smoke():
    result = run_script_smoke("def broken(:\n    pass")

    assert result.passed is False
    assert result.status == "fail"
    assert any(issue.code == "python_syntax_error" for issue in result.issues)


def test_missing_push_to_hub_fails_training_smoke():
    script = VALID_SFT_SCRIPT.replace("    push_to_hub=True,\n", "")

    result = run_script_smoke(script, job_args={"timeout": "8h"})

    assert result.passed is False
    assert any(issue.code == "missing_push_to_hub" for issue in result.blocking_issues)


def test_unrelated_hub_kwargs_do_not_satisfy_training_config_requirements():
    script = """
from trl import SFTConfig

config = SFTConfig(output_dir="out")
unrelated(push_to_hub=True, hub_model_id="user/model")
""".strip()

    result = run_script_smoke(script, job_args={"timeout": "8h"})

    assert result.passed is False
    assert any(issue.code == "missing_push_to_hub" for issue in result.blocking_issues)
    assert any(issue.code == "missing_hub_model_id" for issue in result.blocking_issues)


def test_missing_hub_model_id_fails_training_smoke():
    script = VALID_SFT_SCRIPT.replace('    hub_model_id="user/model",\n', "")

    result = run_script_smoke(script, job_args={"timeout": "8h"})

    assert result.passed is False
    assert any(issue.code == "missing_hub_model_id" for issue in result.blocking_issues)


def test_unsupported_sft_config_max_seq_length_fails_training_smoke():
    script = VALID_SFT_SCRIPT.replace(
        '    hub_model_id="user/model",\n',
        '    hub_model_id="user/model",\n    max_seq_length=512,\n',
    )

    result = run_script_smoke(script, job_args={"timeout": "8h"})

    assert result.passed is False
    assert any(
        issue.code == "unsupported_sft_config_kwarg" for issue in result.blocking_issues
    )


def test_unsupported_sft_config_evaluation_strategy_fails_training_smoke():
    script = VALID_SFT_SCRIPT.replace(
        '    hub_model_id="user/model",\n',
        '    hub_model_id="user/model",\n    evaluation_strategy="steps",\n',
    )

    result = run_script_smoke(script, job_args={"timeout": "8h"})

    assert result.passed is False
    assert any(
        issue.code == "unsupported_sft_config_kwarg"
        and "evaluation_strategy" in issue.message
        and "eval_strategy" in issue.message
        for issue in result.blocking_issues
    )


def test_literal_kwargs_sft_config_evaluation_strategy_fails_training_smoke():
    script = """
from trl import SFTConfig, SFTTrainer

config = SFTConfig(
    **{
        "output_dir": "out",
        "push_to_hub": True,
        "hub_model_id": "user/model",
        "evaluation_strategy": "steps",
    }
)
trainer = SFTTrainer(args=config)
trainer.train()
""".strip()

    result = run_script_smoke(script, job_args={"timeout": "8h"})

    assert result.passed is False
    assert any(
        issue.code == "unsupported_sft_config_kwarg"
        and "evaluation_strategy" in issue.message
        and "eval_strategy" in issue.message
        for issue in result.blocking_issues
    )


def test_unsupported_sft_trainer_tokenizer_fails_training_smoke():
    script = VALID_SFT_SCRIPT.replace(
        "trainer = SFTTrainer(args=config)",
        'trainer = SFTTrainer(args=config, tokenizer="tokenizer")',
    )

    result = run_script_smoke(script, job_args={"timeout": "8h"})

    assert result.passed is False
    assert any(
        issue.code == "unsupported_sft_trainer_kwarg"
        for issue in result.blocking_issues
    )


@pytest.mark.parametrize("kwarg", ["project_name", "run_name"])
def test_unsupported_trackio_init_kwargs_fail_training_smoke(kwarg):
    script = VALID_SFT_SCRIPT.replace(
        'trackio.init(project="phase-5")',
        f'trackio.init(project="phase-5", {kwarg}="stale")',
    )

    result = run_script_smoke(
        script,
        job_args={
            "operation": "run",
            "timeout": "8h",
            "dependencies": ["trl", "trackio"],
        },
    )

    assert result.passed is False
    assert any(
        issue.code == "unsupported_trackio_init_kwarg" and kwarg in issue.message
        for issue in result.blocking_issues
    )


def test_literal_trackio_init_kwargs_fail_training_smoke():
    script = VALID_SFT_SCRIPT.replace(
        'trackio.init(project="phase-5")',
        'trackio.init(**{"project": "phase-5", "project_name": "old", "run_name": "old"})',
    )

    result = run_script_smoke(
        script,
        job_args={
            "operation": "run",
            "timeout": "8h",
            "dependencies": ["trl", "trackio"],
        },
    )

    assert result.passed is False
    assert [issue.code for issue in result.blocking_issues].count(
        "unsupported_trackio_init_kwarg"
    ) == 2


def test_visible_training_without_config_fails_unverified_hub_persistence():
    script = """
from trl import SFTTrainer

trainer = SFTTrainer(model="model")
trainer.train()
""".strip()

    result = run_script_smoke(script, job_args={"timeout": "8h"})

    assert result.passed is False
    assert any(
        issue.code == "hub_persistence_unverified" for issue in result.blocking_issues
    )


def test_literal_kwargs_training_config_satisfies_hub_persistence():
    script = """
from trl import SFTConfig, SFTTrainer

config = SFTConfig(**{"output_dir": "out", "push_to_hub": True, "hub_model_id": "user/model"})
trainer = SFTTrainer(args=config)
trainer.train()
""".strip()

    result = run_script_smoke(
        script,
        job_args={
            "timeout": "8h",
            "dependencies": ["trl", "trackio"],
        },
    )

    assert result.passed is True
    assert result.blocking_issues == []


def test_dynamic_kwargs_training_config_conservatively_fails():
    script = """
from trl import SFTConfig, SFTTrainer

kwargs = {"output_dir": "out", "push_to_hub": True, "hub_model_id": "user/model"}
config = SFTConfig(**kwargs)
trainer = SFTTrainer(args=config)
trainer.train()
""".strip()

    result = run_script_smoke(script, job_args={"timeout": "8h"})

    assert result.passed is False
    assert any(
        issue.code == "hub_persistence_unverified" for issue in result.blocking_issues
    )


def test_default_or_too_short_training_timeout_warns():
    result = run_script_smoke(VALID_SFT_SCRIPT, job_args={"timeout": "30m"})

    assert result.passed is True
    assert result.status == "warn"
    assert any(issue.code == "training_timeout_too_short" for issue in result.warnings)


def test_missing_trackio_config_warns_for_training():
    script = VALID_SFT_SCRIPT.replace("import trackio\n", "").replace(
        'trackio.init(project="phase-5")\n\n',
        "",
    )

    result = run_script_smoke(
        script, job_args={"timeout": "8h", "dependencies": ["trl"]}
    )

    assert result.passed is True
    assert result.status == "warn"
    assert any(issue.code == "missing_trackio" for issue in result.warnings)


def test_explicitly_disabled_reporting_does_not_require_trackio():
    script = (
        VALID_SFT_SCRIPT.replace("import trackio\n", "")
        .replace(
            'trackio.init(project="phase-5")\n\n',
            "",
        )
        .replace(
            '    hub_model_id="user/model",\n',
            '    hub_model_id="user/model",\n    report_to=[],\n',
        )
    )

    result = run_script_smoke(
        script, job_args={"timeout": "8h", "dependencies": ["trl"]}
    )

    assert result.passed is True
    assert result.status == "pass"
    assert not any(issue.code == "missing_trackio" for issue in result.warnings)


def test_error_issues_in_serialized_smoke_metadata_force_failed_status():
    result = coerce_script_smoke_result(
        {
            "status": "pass",
            "is_training_script": True,
            "issues": [
                {
                    "severity": "error",
                    "code": "missing_push_to_hub",
                    "message": "missing push",
                }
            ],
        }
    )

    assert result.passed is False
    assert result.status == "fail"


def test_non_training_python_script_is_not_blocked_by_training_requirements():
    result = run_script_smoke(
        "from transformers import pipeline\nprint(pipeline('sentiment-analysis')('ok'))",
        job_args={"timeout": "30m", "dependencies": ["transformers"]},
    )

    assert result.passed is True
    assert result.is_training_script is False
    assert result.blocking_issues == []
    assert not any(issue.code == "missing_push_to_hub" for issue in result.issues)


@pytest.mark.asyncio
async def test_hf_jobs_approval_guard_blocks_failed_script_smoke(monkeypatch):
    async def fake_estimate_tool_cost(*_args, **_kwargs):
        from agent.core.cost_estimation import CostEstimate

        return CostEstimate(estimated_cost_usd=0.05, billable=True, label="t4-small")

    monkeypatch.setattr(
        "agent.core.approval_policy.estimate_tool_cost",
        fake_estimate_tool_cost,
    )

    decision = await decide_tool_approval(
        "hf_jobs",
        {
            "operation": "run",
            "script": VALID_SFT_SCRIPT.replace("    push_to_hub=True,\n", ""),
            "hardware_flavor": "t4-small",
            "timeout": "8h",
            "dataset_profile": {
                "row_count": 1200,
                "columns": ["messages"],
                "format": "jsonl",
                "sample_rows": [
                    {
                        "messages": [
                            {"role": "user", "content": "Question?"},
                            {"role": "assistant", "content": "Answer."},
                        ]
                    }
                ],
            },
        },
        session=_session(),
    )

    assert decision.approved is False
    assert decision.requires_approval is True
    assert decision.blocked is True
    assert decision.auto_approval_blocked is True
    assert "Script smoke" in decision.block_reason
    assert "push_to_hub" in decision.block_reason


@pytest.mark.asyncio
async def test_approval_recomputes_inline_smoke_instead_of_trusting_metadata(
    monkeypatch,
):
    async def fake_estimate_tool_cost(*_args, **_kwargs):
        from agent.core.cost_estimation import CostEstimate

        return CostEstimate(estimated_cost_usd=0.05, billable=True, label="t4-small")

    monkeypatch.setattr(
        "agent.core.approval_policy.estimate_tool_cost",
        fake_estimate_tool_cost,
    )

    decision = await decide_tool_approval(
        "hf_jobs",
        {
            "operation": "run",
            "script": VALID_SFT_SCRIPT.replace("    push_to_hub=True,\n", ""),
            "script_smoke": {
                "status": "pass",
                "passed": True,
                "is_training_script": True,
                "issues": [],
            },
            "hardware_flavor": "t4-small",
            "timeout": "8h",
            "dataset_profile": {
                "row_count": 1200,
                "columns": ["messages"],
                "format": "jsonl",
                "sample_rows": [
                    {
                        "messages": [
                            {"role": "user", "content": "Question?"},
                            {"role": "assistant", "content": "Answer."},
                        ]
                    }
                ],
            },
        },
        session=_session(),
    )

    assert decision.approved is False
    assert decision.blocked is True
    assert "missing_push_to_hub" in decision.block_reason


@pytest.mark.asyncio
async def test_approval_blocks_remote_training_script_as_unverified(monkeypatch):
    async def fake_estimate_tool_cost(*_args, **_kwargs):
        from agent.core.cost_estimation import CostEstimate

        return CostEstimate(estimated_cost_usd=0.05, billable=True, label="t4-small")

    monkeypatch.setattr(
        "agent.core.approval_policy.estimate_tool_cost",
        fake_estimate_tool_cost,
    )

    decision = await decide_tool_approval(
        "hf_jobs",
        {
            "operation": "run",
            "script": "https://example.com/train.py",
            "script_smoke": {
                "status": "pass",
                "passed": True,
                "is_training_script": True,
                "issues": [],
            },
            "hardware_flavor": "t4-small",
            "timeout": "8h",
            "dataset_profile": {
                "row_count": 1200,
                "columns": ["messages"],
                "format": "jsonl",
                "sample_rows": [
                    {
                        "messages": [
                            {"role": "user", "content": "Question?"},
                            {"role": "assistant", "content": "Answer."},
                        ]
                    }
                ],
            },
        },
        session=_session(),
    )

    assert decision.approved is False
    assert decision.blocked is True
    assert "unverified" in decision.block_reason.lower()


@pytest.mark.asyncio
async def test_approval_blocks_remote_script_with_training_smoke_metadata(
    monkeypatch,
):
    async def fake_estimate_tool_cost(*_args, **_kwargs):
        from agent.core.cost_estimation import CostEstimate

        return CostEstimate(estimated_cost_usd=0.05, billable=True, label="t4-small")

    monkeypatch.setattr(
        "agent.core.approval_policy.estimate_tool_cost",
        fake_estimate_tool_cost,
    )

    decision = await decide_tool_approval(
        "hf_jobs",
        {
            "operation": "run",
            "script": "https://example.com/run.py",
            "script_smoke": {
                "status": "pass",
                "passed": True,
                "is_training_script": True,
                "issues": [],
            },
            "hardware_flavor": "t4-small",
            "timeout": "8h",
            "dataset_profile": {
                "row_count": 1200,
                "columns": ["messages"],
                "format": "jsonl",
                "sample_rows": [
                    {
                        "messages": [
                            {"role": "user", "content": "Question?"},
                            {"role": "assistant", "content": "Answer."},
                        ]
                    }
                ],
            },
        },
        session=_session(),
    )

    assert decision.approved is False
    assert decision.blocked is True
    assert "unverified" in decision.block_reason.lower()


@pytest.mark.asyncio
async def test_approval_blocks_unresolved_training_path_as_unverified(monkeypatch):
    async def fake_estimate_tool_cost(*_args, **_kwargs):
        from agent.core.cost_estimation import CostEstimate

        return CostEstimate(estimated_cost_usd=0.05, billable=True, label="t4-small")

    monkeypatch.setattr(
        "agent.core.approval_policy.estimate_tool_cost",
        fake_estimate_tool_cost,
    )

    decision = await decide_tool_approval(
        "hf_jobs",
        {
            "operation": "run",
            "script": "/app/train.py",
            "hardware_flavor": "t4-small",
            "timeout": "8h",
            "dataset_profile": {
                "row_count": 1200,
                "columns": ["messages"],
                "format": "jsonl",
                "sample_rows": [
                    {
                        "messages": [
                            {"role": "user", "content": "Question?"},
                            {"role": "assistant", "content": "Answer."},
                        ]
                    }
                ],
            },
        },
        session=_session(),
    )

    assert decision.approved is False
    assert decision.blocked is True
    assert "unverified" in decision.block_reason.lower()


@pytest.mark.asyncio
async def test_hf_jobs_handler_blocks_failed_one_line_inline_smoke(monkeypatch):
    from agent.tools import jobs_tool

    class FakeProvider:
        async def execute_tool(self, *_args, **_kwargs):  # pragma: no cover - guard
            raise AssertionError("provider should not be called when smoke fails")

    class FakeRegistry:
        def require_executable(self, provider_id):
            assert provider_id == "hf-jobs"
            return FakeProvider()

    monkeypatch.setattr(
        jobs_tool,
        "get_cloud_provider_registry",
        lambda: FakeRegistry(),
    )

    output, ok = await jobs_tool.hf_jobs_handler(
        {
            "operation": "run",
            "script": "SFTConfig(output_dir='out')",
            "timeout": "8h",
        },
        session=SimpleNamespace(hf_token=None, sandbox=None),
        tool_call_id="tc-smoke",
    )

    assert ok is False
    assert "submission blocked" in output
    assert "missing_push_to_hub" in output


@pytest.mark.asyncio
async def test_hf_jobs_handler_blocks_remote_script_with_training_smoke_metadata(
    monkeypatch,
):
    from agent.tools import jobs_tool

    class FakeProvider:
        async def execute_tool(self, *_args, **_kwargs):  # pragma: no cover - guard
            raise AssertionError(
                "provider should not be called when smoke is unverified"
            )

    class FakeRegistry:
        def require_executable(self, provider_id):
            assert provider_id == "hf-jobs"
            return FakeProvider()

    monkeypatch.setattr(
        jobs_tool,
        "get_cloud_provider_registry",
        lambda: FakeRegistry(),
    )

    output, ok = await jobs_tool.hf_jobs_handler(
        {
            "operation": "run",
            "script": "https://example.com/run.py",
            "script_smoke": {
                "status": "pass",
                "passed": True,
                "is_training_script": True,
                "issues": [],
            },
            "timeout": "8h",
        },
        session=SimpleNamespace(hf_token=None, sandbox=None),
        tool_call_id="tc-smoke",
    )

    assert ok is False
    assert "submission blocked" in output
    assert "unverified" in output.lower()


@pytest.mark.asyncio
async def test_hf_jobs_execute_blocks_failed_inline_training_smoke_before_api_call():
    class FakeApi:
        def run_job(self, **_kwargs):  # pragma: no cover - guard
            raise AssertionError("run_job should not be called when smoke fails")

    tool = HfJobsTool()
    tool.api = FakeApi()

    result = await tool.execute(
        {
            "operation": "run",
            "script": VALID_SFT_SCRIPT.replace(
                '    hub_model_id="user/model",\n',
                '    hub_model_id="user/model",\n    max_seq_length=512,\n',
            ),
            "dependencies": ["trl", "trackio"],
            "hardware_flavor": "t4-small",
            "timeout": "8h",
        }
    )

    assert result["isError"] is True
    assert "submission blocked" in result["formatted"]
    assert "max_seq_length" in result["formatted"]


def test_one_line_inline_python_script_is_wrapped_for_execution():
    command = _resolve_uv_command("print('hello')", with_deps=["rich"])

    assert command[:2] == ["/bin/sh", "-lc"]
    assert "base64 -d | uv run --with rich -" in command[2]


@pytest.mark.asyncio
async def test_hf_jobs_streamed_and_final_logs_are_redacted():
    class FakeStatus:
        stage = "COMPLETED"

    class FakeJobInfo:
        status = FakeStatus()

    class FakeApi:
        def fetch_job_logs(self, **_kwargs):
            yield "HF_TOKEN=hf_" + "a" * 30
            yield "done"

        def inspect_job(self, **_kwargs):
            return FakeJobInfo()

    streamed_logs = []

    async def capture_log(log):
        streamed_logs.append(log)

    tool = HfJobsTool(log_callback=capture_log)
    tool.api = FakeApi()

    final_status, all_logs = await tool._wait_for_job_completion("job-1")

    assert final_status == "COMPLETED"
    assert streamed_logs == ["HF_TOKEN=[REDACTED_HF_TOKEN]", "done"]
    assert all_logs == ["HF_TOKEN=[REDACTED_HF_TOKEN]", "done"]
