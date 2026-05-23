"""Phase 5 tests for cost estimation and approval policy."""

from types import SimpleNamespace

import pytest
from litellm import ChatCompletionMessageToolCall as ToolCall

from agent.config import Config
from agent.core import cost_estimation
from agent.core.agent_loop import _needs_approval, _prepare_approval_event_data
from agent.core.approval_policy import decide_tool_approval
from agent.main import _print_approval_metadata


def _config(**overrides):
    return Config(model_name="test-model", **overrides)


def _session(**overrides):
    values = {
        "config": _config(),
        "sandbox": None,
        "auto_approval_estimated_spend_usd": 0.0,
        "auto_approval_cost_cap_usd": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parse_timeout_hours_common_units_and_invalid_values():
    assert cost_estimation.parse_timeout_hours(None) == 0.5
    assert cost_estimation.parse_timeout_hours("") == 0.5
    assert cost_estimation.parse_timeout_hours("30m") == 0.5
    assert cost_estimation.parse_timeout_hours("3h") == 3
    assert cost_estimation.parse_timeout_hours("1d") == 24
    assert cost_estimation.parse_timeout_hours(3600) == 1
    assert cost_estimation.parse_timeout_hours(True) is None
    assert cost_estimation.parse_timeout_hours("not-a-duration") is None
    assert cost_estimation.parse_timeout_hours("-1h") is None


@pytest.mark.asyncio
async def test_estimate_hf_job_cost_uses_known_flavor(monkeypatch):
    async def fake_catalog():
        return {"a100-large": 4.0}

    monkeypatch.setattr(cost_estimation, "hf_jobs_price_catalog", fake_catalog)

    estimate = await cost_estimation.estimate_hf_job_cost(
        {"operation": "run", "hardware_flavor": "a100-large", "timeout": "8h"}
    )

    assert estimate.estimated_cost_usd == 32.0
    assert estimate.billable is True
    assert estimate.block_reason is None
    assert estimate.label == "a100-large"


@pytest.mark.asyncio
async def test_estimate_hf_job_cost_unknown_flavor_requires_manual(monkeypatch):
    async def fake_catalog():
        return {}

    monkeypatch.setattr(cost_estimation, "hf_jobs_price_catalog", fake_catalog)

    estimate = await cost_estimation.estimate_hf_job_cost(
        {"operation": "run", "hardware_flavor": "mystery-gpu", "timeout": "30m"}
    )

    assert estimate.estimated_cost_usd is None
    assert estimate.billable is True
    assert "No price" in estimate.block_reason
    assert estimate.label == "mystery-gpu"


@pytest.mark.asyncio
async def test_estimate_sandbox_cost_for_existing_cpu_and_gpu():
    existing = await cost_estimation.estimate_sandbox_cost(
        {"hardware": "a100-large"},
        session=_session(sandbox=object()),
    )
    cpu = await cost_estimation.estimate_sandbox_cost({"hardware": "cpu-basic"})
    gpu = await cost_estimation.estimate_sandbox_cost({"hardware": "t4-small"})

    assert existing.estimated_cost_usd == 0.0
    assert existing.billable is False
    assert cpu.estimated_cost_usd == 0.0
    assert cpu.billable is False
    assert gpu.estimated_cost_usd == 0.6
    assert gpu.billable is True


@pytest.mark.asyncio
async def test_low_cost_auto_approval_when_yolo_and_within_cap(monkeypatch):
    async def fake_estimate_tool_cost(*_args, **_kwargs):
        return cost_estimation.CostEstimate(
            estimated_cost_usd=0.05, billable=True, label="cpu-basic"
        )

    monkeypatch.setattr(
        "agent.core.approval_policy.estimate_tool_cost", fake_estimate_tool_cost
    )
    session = _session(
        config=_config(yolo_mode=True),
        auto_approval_cost_cap_usd=1.0,
    )

    decision = await decide_tool_approval(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "cpu-basic", "timeout": "30m"},
        session=session,
    )

    assert decision.approved is True
    assert decision.requires_approval is False
    assert decision.auto_approved is True
    assert decision.estimated_cost_usd == 0.05
    assert decision.remaining_cap_usd == 1.0


@pytest.mark.asyncio
async def test_billable_job_without_cap_requires_manual_approval_in_yolo(monkeypatch):
    async def fake_estimate_tool_cost(*_args, **_kwargs):
        return cost_estimation.CostEstimate(
            estimated_cost_usd=0.05, billable=True, label="t4-small"
        )

    monkeypatch.setattr(
        "agent.core.approval_policy.estimate_tool_cost", fake_estimate_tool_cost
    )
    session = _session(config=_config(yolo_mode=True))

    decision = await decide_tool_approval(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "t4-small", "timeout": "30m"},
        session=session,
    )

    assert decision.approved is False
    assert decision.requires_approval is True
    assert decision.auto_approval_blocked is True
    assert decision.estimated_cost_usd == 0.05
    assert decision.remaining_cap_usd == 0.0
    assert "exceeds remaining" in decision.block_reason


@pytest.mark.asyncio
async def test_high_cost_over_cap_requires_manual_approval(monkeypatch):
    async def fake_estimate_tool_cost(*_args, **_kwargs):
        return cost_estimation.CostEstimate(
            estimated_cost_usd=2.0, billable=True, label="a100-large"
        )

    monkeypatch.setattr(
        "agent.core.approval_policy.estimate_tool_cost", fake_estimate_tool_cost
    )
    session = _session(
        config=_config(yolo_mode=True),
        auto_approval_cost_cap_usd=1.0,
        auto_approval_estimated_spend_usd=0.25,
    )

    decision = await decide_tool_approval(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "a100-large", "timeout": "1h"},
        session=session,
    )

    assert decision.approved is False
    assert decision.requires_approval is True
    assert decision.auto_approval_blocked is True
    assert decision.blocked is False
    assert decision.remaining_cap_usd == 0.75
    assert "exceeds remaining" in decision.block_reason


@pytest.mark.asyncio
async def test_unknown_cost_billable_action_requires_manual_approval(monkeypatch):
    async def fake_estimate_tool_cost(*_args, **_kwargs):
        return cost_estimation.CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason="No price is available.",
            label="mystery-gpu",
        )

    monkeypatch.setattr(
        "agent.core.approval_policy.estimate_tool_cost", fake_estimate_tool_cost
    )
    session = _session(
        config=_config(yolo_mode=True),
        auto_approval_cost_cap_usd=10.0,
    )

    decision = await decide_tool_approval(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "mystery-gpu", "timeout": "30m"},
        session=session,
    )

    assert decision.requires_approval is True
    assert decision.unknown_cost is True
    assert decision.auto_approval_blocked is True
    assert decision.block_reason == "No price is available."


@pytest.mark.asyncio
async def test_scheduled_jobs_always_require_manual_approval_even_in_yolo():
    decision = await decide_tool_approval(
        "hf_jobs",
        {"operation": "scheduled run", "hardware_flavor": "cpu-basic"},
        session=_session(config=_config(yolo_mode=True)),
    )

    assert decision.approved is False
    assert decision.requires_approval is True
    assert decision.auto_approval_blocked is True
    assert "Scheduled" in decision.block_reason


def test_legacy_needs_approval_preserves_scheduled_job_manual_gate_in_yolo():
    assert _needs_approval(
        "hf_jobs",
        {"operation": "scheduled run", "hardware_flavor": "cpu-basic"},
        _config(yolo_mode=True),
    )


@pytest.mark.asyncio
async def test_approval_event_shape_remains_compatible(monkeypatch):
    async def fake_resolve_sandbox_script(_sandbox, _script):
        return "print('resolved')", None

    monkeypatch.setattr(
        "agent.core.agent_loop.resolve_sandbox_script",
        fake_resolve_sandbox_script,
        raising=False,
    )
    tool_call = ToolCall(
        id="tc1",
        type="function",
        function={
            "name": "hf_jobs",
            "arguments": '{"operation": "run", "script": "/app/train.py"}',
        },
    )
    decision = SimpleNamespace(
        auto_approval_blocked=True,
        block_reason="Estimated cost exceeds remaining cap.",
        estimated_cost_usd=2.0,
        remaining_cap_usd=1.0,
    )

    event_data = await _prepare_approval_event_data(
        [
            (
                tool_call,
                "hf_jobs",
                {"operation": "run", "script": "/app/train.py"},
                decision,
            )
        ],
        session=_session(),
    )

    assert event_data["count"] == 1
    assert event_data["tools"][0]["tool"] == "hf_jobs"
    assert event_data["tools"][0]["tool_call_id"] == "tc1"
    assert event_data["tools"][0]["arguments"]["script"] == "print('resolved')"
    assert event_data["auto_approval_blocked"] is True
    assert event_data["block_reason"] == "Estimated cost exceeds remaining cap."


def test_cli_approval_metadata_is_printed_for_manual_prompt(capsys):
    _print_approval_metadata(
        {
            "auto_approval_blocked": True,
            "block_reason": "Estimated cost $2.00 exceeds remaining cap $1.00.",
            "estimated_cost_usd": 2.0,
            "remaining_cap_usd": 1.0,
        }
    )

    output = capsys.readouterr().out
    assert "Auto-approval blocked" in output
    assert "Estimated cost $2.00 exceeds remaining cap $1.00." in output
    assert "Estimated cost: $2.00" in output
    assert "Remaining auto-approval cap: $1.00" in output
