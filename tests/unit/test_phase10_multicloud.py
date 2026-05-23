"""Phase 10 tests for multicloud plan-only providers."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import domain_templates  # noqa: E402
import session_manager as session_manager_module  # noqa: E402
from agent.core.cost_estimation import CostEstimate, estimate_tool_cost  # noqa: E402
from agent.core.redact import scrub  # noqa: E402


def test_phase10_registry_exposes_active_plan_only_cloud_providers():
    from agent.core.cloud_providers import get_cloud_provider_registry

    registry = get_cloud_provider_registry()

    for provider_id, label in (
        ("aws", "AWS"),
        ("azure", "Azure"),
        ("gcp", "Google Cloud"),
    ):
        provider = registry.require(provider_id)
        assert provider.enabled is True
        assert provider.executable is False
        assert provider.display_name == label
        assert provider.disabled_reason is None


@pytest.mark.asyncio
async def test_phase10_plan_only_provider_returns_training_plan_and_readiness():
    from agent.core.cloud_providers import ProviderContext, get_cloud_provider_registry

    provider = get_cloud_provider_registry().require("aws")
    result = await provider.execute_tool(
        {
            "operation": "plan",
            "script": "train.py",
            "hardware_flavor": "a10g-large",
            "timeout": "2h",
            "env": {"AWS_SECRET_ACCESS_KEY": "fake-secret"},
        },
        ProviderContext(),
    )

    assert result["isError"] is False
    assert result["totalResults"] == 1
    assert "AWS plan-only training plan" in result["formatted"]
    assert "No AWS resources were created" in result["formatted"]
    assert "Credential readiness" in result["formatted"]
    assert "fake-secret" not in result["formatted"]


@pytest.mark.asyncio
async def test_phase10_plan_only_submit_is_blocked_without_cloud_spend():
    from agent.core.cloud_providers import ProviderContext, get_cloud_provider_registry

    provider = get_cloud_provider_registry().require("gcp")
    result = await provider.execute_tool(
        {"operation": "submit", "command": ["python", "train.py"]},
        ProviderContext(),
    )

    assert result["isError"] is True
    assert "Real Google Cloud execution is not enabled" in result["formatted"]
    assert "No Google Cloud resources were created" in result["formatted"]


@pytest.mark.asyncio
async def test_phase10_hf_jobs_handler_routes_to_selected_session_provider(monkeypatch):
    from agent.tools import jobs_tool

    calls = []

    class FakeProvider:
        provider_id = "aws"

        async def execute_tool(self, arguments, context):
            calls.append((arguments, context))
            return {
                "formatted": "aws plan from fake provider",
                "totalResults": 1,
                "resultsShared": 1,
                "isError": False,
            }

    class FakeRegistry:
        def require(self, provider_id):
            assert provider_id == "aws"
            return FakeProvider()

    monkeypatch.setattr(
        jobs_tool, "get_cloud_provider_registry", lambda: FakeRegistry()
    )

    session = SimpleNamespace(
        hf_token="token",
        sandbox=None,
        provider_id="aws",
    )
    output, ok = await jobs_tool.hf_jobs_handler(
        {"operation": "plan", "script": "train.py"},
        session=session,
        tool_call_id="tc-phase10",
    )

    assert ok is True
    assert output == "aws plan from fake provider"
    assert calls[0][0]["operation"] == "plan"
    assert calls[0][1].provider_id == "aws"
    assert calls[0][1].tool_name == "hf_jobs"


@pytest.mark.asyncio
async def test_phase10_plan_only_cost_estimate_is_not_billable():
    session = SimpleNamespace(provider_id="azure")

    estimate = await estimate_tool_cost(
        "hf_jobs",
        {"operation": "run", "script": "train.py", "timeout": "4h"},
        session=session,
    )

    assert estimate == CostEstimate(
        estimated_cost_usd=0.0,
        billable=False,
        block_reason=None,
        label="azure-plan-only",
    )


@pytest.mark.asyncio
async def test_phase10_real_submit_requires_manual_block_for_plan_only_provider():
    session = SimpleNamespace(provider_id="aws")

    estimate = await estimate_tool_cost(
        "hf_jobs",
        {"operation": "submit", "script": "train.py"},
        session=session,
    )

    assert estimate.estimated_cost_usd is None
    assert estimate.billable is True
    assert "not enabled" in estimate.block_reason
    assert estimate.label == "aws-submit"


def test_phase10_workflow_context_tells_non_hf_providers_to_plan_only():
    context = domain_templates.render_workflow_context(
        {"domain_id": "call-center", "provider_id": "aws"}
    )

    assert "- Compute provider: AWS" in context
    assert "plan-only preview" in context
    assert "Do not create AWS resources" in context
    assert "Phase 6 approval-before-spend fast path" not in context


def test_phase10_redacts_cloud_credential_like_values():
    payload = {
        "AWS_ACCESS_KEY_ID": "AKIAABCDEFGHIJKLMNOP",
        "aws_secret_access_key": "abcdefghijklmnopqrstuvwxyz1234567890",
        "AZURE_TENANT_ID": "11111111-2222-3333-4444-555555555555",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON": '{"private_key":"-----BEGIN PRIVATE KEY-----abc-----END PRIVATE KEY-----"}',
    }

    scrubbed = scrub(payload)

    assert scrubbed["AWS_ACCESS_KEY_ID"] == "[REDACTED_SECRET]"
    assert scrubbed["aws_secret_access_key"] == "[REDACTED_SECRET]"
    assert scrubbed["AZURE_TENANT_ID"] == "[REDACTED_SECRET]"
    assert scrubbed["GOOGLE_APPLICATION_CREDENTIALS_JSON"] == "[REDACTED_SECRET]"


def test_phase10_session_metadata_updates_runtime_provider(monkeypatch):
    manager = session_manager_module.SessionManager.__new__(
        session_manager_module.SessionManager
    )
    manager._lock = asyncio.Lock()
    agent_session = SimpleNamespace(
        is_active=True,
        domain_id="generic",
        provider_id="hf-jobs",
        dataset_repo=None,
        session=SimpleNamespace(provider_id="hf-jobs"),
    )
    manager.sessions = {"session-1": agent_session}

    async def fake_persist_session_snapshot(_agent_session):
        return None

    monkeypatch.setattr(
        manager,
        "persist_session_snapshot",
        fake_persist_session_snapshot,
    )

    ok = asyncio.run(
        manager.update_session_metadata(
            "session-1",
            domain_id="itr",
            provider_id="gcp",
            dataset_repo="org/data",
        )
    )

    assert ok is True
    assert agent_session.provider_id == "gcp"
    assert agent_session.session.provider_id == "gcp"
