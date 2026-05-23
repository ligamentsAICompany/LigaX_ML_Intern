"""Phase 7 tests for the cloud provider abstraction."""

from types import SimpleNamespace

import pytest

from agent.core.cost_estimation import CostEstimate


def test_cloud_provider_registry_exposes_hf_jobs_and_plan_only_providers():
    from agent.core.cloud_providers import get_cloud_provider_registry

    registry = get_cloud_provider_registry()

    hf_provider = registry.require("hf-jobs")
    assert hf_provider.provider_id == "hf-jobs"
    assert hf_provider.display_name == "Hugging Face Jobs"
    assert hf_provider.enabled is True
    assert hf_provider.executable is True

    for provider_id in ("aws", "azure", "gcp"):
        provider = registry.require(provider_id)
        assert provider.enabled is True
        assert provider.executable is False
        assert provider.disabled_reason is None


@pytest.mark.asyncio
async def test_hf_provider_supplies_job_cost_estimates(monkeypatch):
    from agent.core.cloud_providers import get_cloud_provider_registry

    async def fake_catalog():
        return {"t4-small": 0.6}

    monkeypatch.setattr(
        "agent.core.cloud_providers.hf_jobs_price_catalog", fake_catalog
    )

    provider = get_cloud_provider_registry().require("hf-jobs")
    estimate = await provider.estimate_cost(
        {"operation": "run", "hardware_flavor": "t4-small", "timeout": "30m"}
    )

    assert estimate == CostEstimate(
        estimated_cost_usd=0.3,
        billable=True,
        block_reason=None,
        label="t4-small",
    )


@pytest.mark.asyncio
async def test_hf_jobs_handler_delegates_to_provider(monkeypatch):
    from agent.tools import jobs_tool

    calls = []

    class FakeProvider:
        provider_id = "hf-jobs"

        async def execute_tool(self, arguments, context):
            calls.append((arguments, context))
            return {
                "formatted": "provider handled ps",
                "totalResults": 1,
                "resultsShared": 1,
            }

    class FakeRegistry:
        def require_executable(self, provider_id):
            assert provider_id == "hf-jobs"
            return FakeProvider()

    monkeypatch.setenv("HF_NAMESPACE", "tester")
    monkeypatch.setattr(
        jobs_tool, "get_cloud_provider_registry", lambda: FakeRegistry()
    )

    session = SimpleNamespace(hf_token="token", sandbox=None)
    output, ok = await jobs_tool.hf_jobs_handler(
        {"operation": "ps"}, session=session, tool_call_id="tc1"
    )

    assert ok is True
    assert output == "provider handled ps"
    assert calls[0][0] == {"operation": "ps"}
    assert calls[0][1].tool_name == "hf_jobs"
    assert calls[0][1].tool_call_id == "tc1"
    assert calls[0][1].hf_token == "token"


@pytest.mark.asyncio
async def test_cancel_cleanup_uses_provider_abstraction(monkeypatch):
    from agent.core import agent_loop

    cancelled = []

    class FakeProvider:
        async def cancel_jobs(self, job_ids, context):
            cancelled.extend(job_ids)
            assert context.hf_token == "token"

    class FakeRegistry:
        def require_executable(self, provider_id):
            assert provider_id == "hf-jobs"
            return FakeProvider()

    session = SimpleNamespace(
        sandbox=None,
        hf_token="token",
        _running_job_ids={"job-1", "job-2"},
    )
    monkeypatch.setattr(
        agent_loop, "get_cloud_provider_registry", lambda: FakeRegistry(), raising=False
    )

    await agent_loop._cleanup_on_cancel(session)

    assert sorted(cancelled) == ["job-1", "job-2"]
    assert session._running_job_ids == set()
