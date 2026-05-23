"""Phase 11 tests for operational hardening and admin safety tools."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import main  # noqa: E402
from agent.core.operational_hardening import (  # noqa: E402
    build_kpi_snapshot,
    build_provider_health_snapshot,
    sweep_orphan_jobs,
)
from agent.core.session_persistence import LocalSessionStore  # noqa: E402
from session_manager import AgentSession, SessionManager  # noqa: E402


def test_provider_health_snapshot_reports_readiness_without_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_" + "a" * 40)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAABCDEFGHIJKLMNOP")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    snapshot = build_provider_health_snapshot(hf_token="hf_" + "b" * 40)

    rendered = str(snapshot)
    assert "hf_" + "a" * 40 not in rendered
    assert "hf_" + "b" * 40 not in rendered
    assert "AKIAABCDEFGHIJKLMNOP" not in rendered
    assert snapshot["hf_token"]["configured"] is True
    assert snapshot["providers"]["hf-jobs"]["ready"] is True
    assert snapshot["providers"]["aws"]["ready"] is False
    assert "AWS_SECRET_ACCESS_KEY" in snapshot["providers"]["aws"]["missing"]


def test_provider_health_route_and_security_headers_are_registered():
    response = TestClient(main.app).get(
        "/api/health/providers",
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "providers" in body


@pytest.mark.asyncio
async def test_kpi_snapshot_counts_sessions_events_providers_and_jobs(tmp_path):
    store = LocalSessionStore(tmp_path)
    await store.init()
    await store.save_snapshot(
        session_id="active",
        user_id="owner",
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        runtime_state="processing",
        status="active",
        provider_id="hf-jobs",
        running_job_ids=["job-active"],
    )
    await store.append_event("active", "approval_required", {"tool": "hf_jobs"})
    await store.save_snapshot(
        session_id="ended",
        user_id="owner",
        model="test-model",
        messages=[],
        runtime_state="ended",
        status="ended",
        provider_id="aws",
        running_job_ids=["job-ended"],
    )
    await store.append_event("ended", "error", {"error": "boom"})
    await store.soft_delete_session("ended")

    snapshot = await build_kpi_snapshot(store, live_sessions=1)

    assert snapshot["sessions"]["active"] == 1
    assert snapshot["sessions"]["ended"] == 1
    assert snapshot["sessions"]["deleted"] == 1
    assert snapshot["jobs"]["tracked_running_job_ids"] == 2
    assert snapshot["providers"]["hf-jobs"] == 1
    assert snapshot["providers"]["aws"] == 1
    assert snapshot["events"]["total"] == 2
    assert snapshot["events"]["errors"] == 1
    assert snapshot["events"]["approval_required"] == 1


@pytest.mark.asyncio
async def test_orphan_sweep_dry_run_then_apply_only_cancels_safe_candidates(
    tmp_path, monkeypatch
):
    store = LocalSessionStore(tmp_path)
    await store.init()
    await store.save_snapshot(
        session_id="active",
        user_id="owner",
        model="test-model",
        messages=[],
        runtime_state="processing",
        status="active",
        provider_id="hf-jobs",
        running_job_ids=["job-active"],
    )
    await store.save_snapshot(
        session_id="ended",
        user_id="owner",
        model="test-model",
        messages=[],
        runtime_state="ended",
        status="ended",
        provider_id="hf-jobs",
        running_job_ids=["job-ended"],
    )
    await store.save_snapshot(
        session_id="deleted",
        user_id="owner",
        model="test-model",
        messages=[],
        runtime_state="idle",
        status="active",
        provider_id="hf-jobs",
        running_job_ids=["job-deleted"],
    )
    await store.soft_delete_session("deleted")

    cancelled: list[str] = []

    class FakeProvider:
        async def cancel_jobs(self, job_ids, context):
            cancelled.extend(job_ids)
            assert context.tool_name == "phase11_orphan_sweep"

    class FakeRegistry:
        def require_executable(self, provider_id):
            assert provider_id == "hf-jobs"
            return FakeProvider()

    monkeypatch.setattr(
        "agent.core.operational_hardening.get_cloud_provider_registry",
        lambda: FakeRegistry(),
    )

    dry_run = await sweep_orphan_jobs(store, apply=False, hf_token="token")
    assert sorted(item["job_id"] for item in dry_run["candidates"]) == [
        "job-deleted",
        "job-ended",
    ]
    assert dry_run["applied"] is False
    assert cancelled == []

    applied = await sweep_orphan_jobs(store, apply=True, hf_token="token")
    assert applied["applied"] is True
    assert sorted(cancelled) == ["job-deleted", "job-ended"]

    active = await store.load_session("active")
    ended = await store.load_session("ended")
    deleted = await store.load_session("deleted", include_deleted=True)
    assert active["metadata"]["running_job_ids"] == ["job-active"]
    assert ended["metadata"]["running_job_ids"] == []
    assert deleted["metadata"]["running_job_ids"] == []


@pytest.mark.asyncio
async def test_interrupt_delete_and_close_cancel_tracked_jobs_with_provider(
    monkeypatch,
):
    cancelled: list[str] = []

    class FakeProvider:
        async def cancel_jobs(self, job_ids, context):
            cancelled.extend(job_ids)
            assert context.hf_token == "token"

    class FakeRegistry:
        def require_executable(self, provider_id):
            assert provider_id == "hf-jobs"
            return FakeProvider()

    monkeypatch.setattr(
        "session_manager.get_cloud_provider_registry",
        lambda: FakeRegistry(),
        raising=False,
    )

    async def run_until_cancelled():
        await asyncio.Event().wait()

    manager = SessionManager()
    manager.persistence_store = SimpleNamespace(enabled=False, close=lambda: None)
    session = SimpleNamespace(
        _running_job_ids={"job-1"},
        hf_token="token",
        provider_id="hf-jobs",
        is_running=True,
        cancel=lambda: None,
        sandbox=None,
        config=SimpleNamespace(model_name="test-model"),
        context_manager=SimpleNamespace(items=[]),
        pending_approval=None,
        turn_count=0,
    )
    agent_session = AgentSession(
        session_id="session-1",
        session=session,
        tool_router=SimpleNamespace(),
        submission_queue=asyncio.Queue(),
        user_id="owner",
        hf_token="token",
    )
    agent_session.task = asyncio.create_task(run_until_cancelled())
    manager.sessions["session-1"] = agent_session

    assert await manager.interrupt("session-1") is True
    assert session._running_job_ids == set()

    session._running_job_ids = {"job-2"}
    assert await manager.delete_session("session-1") is True

    session._running_job_ids = {"job-3"}
    manager.sessions["session-1"] = agent_session
    await manager.close()

    assert sorted(cancelled) == ["job-1", "job-2", "job-3"]
