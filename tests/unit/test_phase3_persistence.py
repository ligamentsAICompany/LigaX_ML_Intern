"""Phase 3 tests for durable session persistence and replayable SSE."""

import asyncio
import sys
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from litellm import Message

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import session_manager as session_manager_module  # noqa: E402
from agent.core.session_persistence import LocalSessionStore, NoopSessionStore  # noqa: E402
from routes import agent as agent_routes  # noqa: E402
from session_manager import AgentSession, Operation, SessionManager  # noqa: E402


@pytest.mark.asyncio
async def test_local_session_store_persists_snapshots_and_replays_events(tmp_path):
    store = LocalSessionStore(tmp_path)
    await store.init()

    await store.save_snapshot(
        session_id="s1",
        user_id="u1",
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        runtime_state="processing",
        turn_count=2,
        pending_approval=[{"tool": "hf_jobs", "tool_call_id": "tc1", "arguments": {}}],
        claude_counted=True,
    )
    assert await store.append_event("s1", "processing", {"message": "start"}) == 1
    assert await store.append_event("s1", "turn_complete", {}) == 2

    reloaded = LocalSessionStore(tmp_path)
    await reloaded.init()

    loaded = await reloaded.load_session("s1")
    assert loaded is not None
    assert loaded["metadata"]["user_id"] == "u1"
    assert loaded["metadata"]["runtime_state"] == "processing"
    assert loaded["metadata"]["message_count"] == 1
    assert loaded["messages"] == [{"role": "user", "content": "hello"}]

    listed = await reloaded.list_sessions("u1")
    assert [row["session_id"] for row in listed] == ["s1"]
    assert await reloaded.load_events_after("s1", 1) == [
        {"seq": 2, "event_type": "turn_complete", "data": {}}
    ]


@pytest.mark.asyncio
async def test_noop_session_store_keeps_dev_mode_simple():
    store = NoopSessionStore()

    await store.init()
    await store.save_snapshot(
        session_id="s1",
        user_id="u1",
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert store.enabled is False
    assert await store.load_session("s1") is None
    assert await store.list_sessions("u1") == []
    assert await store.append_event("s1", "ready", {}) is None
    assert await store.load_events_after("s1", 0) == []


class _FakeToolRouter:
    pass


class _FakeSession:
    def __init__(self, *, session_id: str, model: str):
        self.session_id = session_id
        self.config = SimpleNamespace(model_name=model)
        self.context_manager = SimpleNamespace(
            items=[Message(role="system", content="fresh system prompt")]
        )
        self.pending_approval = None
        self.turn_count = 0
        self.is_running = True
        self.hf_token = None
        self.sandbox = None
        self._running_job_ids = set()

    def update_model(self, model: str) -> None:
        self.config.model_name = model


@pytest.mark.asyncio
async def test_session_manager_does_not_restore_ended_sessions(tmp_path, monkeypatch):
    store = LocalSessionStore(tmp_path)
    await store.init()
    await store.save_snapshot(
        session_id="ended",
        user_id="owner",
        model="restored-model",
        messages=[{"role": "user", "content": "done"}],
        runtime_state="ended",
        status="ended",
    )

    manager = SessionManager()
    manager.persistence_store = store

    def fail_create_session_sync(**_kwargs):
        raise AssertionError("ended persisted sessions must not be restarted")

    monkeypatch.setattr(manager, "_create_session_sync", fail_create_session_sync)

    assert await manager.ensure_session_loaded("ended", "owner") is None
    assert await manager.submit_user_input("ended", "hello") is False
    assert "ended" not in manager.sessions


@pytest.mark.asyncio
async def test_shutdown_session_does_not_hold_lock_while_waiting_for_task(
    tmp_path, monkeypatch
):
    store = LocalSessionStore(tmp_path)
    await store.init()
    manager = SessionManager()
    manager.persistence_store = store

    session = _FakeSession(session_id="active", model="test-model")
    submission_queue = asyncio.Queue()
    agent_session = AgentSession(
        session_id="active",
        session=session,
        tool_router=_FakeToolRouter(),
        submission_queue=submission_queue,
        user_id="owner",
    )
    assert agent_session.created_at.tzinfo is timezone.utc
    manager.sessions["active"] = agent_session
    await manager.persist_session_snapshot(agent_session, runtime_state="idle")

    async def finalizes_under_manager_lock():
        submission = await submission_queue.get()
        assert isinstance(submission.operation, Operation)
        async with manager._lock:
            agent_session.is_active = False
            await manager.persist_session_snapshot(
                agent_session,
                runtime_state="ended",
                status="ended",
            )

    original_wait_for = asyncio.wait_for

    async def shorten_shutdown_wait(awaitable, timeout=None):
        if timeout == 5.0:
            timeout = 0.05
        return await original_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(
        session_manager_module.asyncio, "wait_for", shorten_shutdown_wait
    )
    agent_session.task = asyncio.create_task(finalizes_under_manager_lock())

    assert await manager.shutdown_session("active") is True

    loaded = await store.load_session("active")
    assert loaded is not None
    assert loaded["metadata"]["status"] == "ended"
    assert loaded["metadata"]["runtime_state"] == "ended"


@pytest.mark.asyncio
async def test_set_session_model_rejects_inactive_memory_session_and_keeps_ended(
    tmp_path, monkeypatch
):
    store = LocalSessionStore(tmp_path)
    await store.init()
    await store.save_snapshot(
        session_id="inactive",
        user_id="owner",
        model="old-model",
        messages=[{"role": "user", "content": "done"}],
        runtime_state="ended",
        status="ended",
    )

    manager = SessionManager()
    manager.persistence_store = store
    manager.sessions["inactive"] = AgentSession(
        session_id="inactive",
        session=_FakeSession(session_id="inactive", model="old-model"),
        tool_router=_FakeToolRouter(),
        submission_queue=asyncio.Queue(),
        user_id="owner",
        is_active=False,
    )
    monkeypatch.setattr(agent_routes, "session_manager", manager)

    with pytest.raises(agent_routes.HTTPException) as exc:
        await agent_routes.set_session_model(
            "inactive",
            {"model": agent_routes.AVAILABLE_MODELS[0]["id"]},
            SimpleNamespace(headers={}, cookies={}),
            {"user_id": "owner", "username": "owner"},
        )

    assert exc.value.status_code == 404
    loaded = await store.load_session("inactive")
    assert loaded is not None
    assert loaded["metadata"]["status"] == "ended"
    assert loaded["metadata"]["runtime_state"] == "ended"
    assert loaded["metadata"]["model"] == "old-model"


@pytest.mark.asyncio
async def test_delete_session_allows_owner_to_soft_delete_ended_persisted_session(
    tmp_path, monkeypatch
):
    store = LocalSessionStore(tmp_path)
    await store.init()
    await store.save_snapshot(
        session_id="ended",
        user_id="owner",
        model="test-model",
        messages=[{"role": "user", "content": "done"}],
        runtime_state="ended",
        status="ended",
    )

    manager = SessionManager()
    manager.persistence_store = store
    monkeypatch.setattr(agent_routes, "session_manager", manager)

    with pytest.raises(agent_routes.HTTPException) as exc:
        await agent_routes.delete_session(
            "ended",
            SimpleNamespace(headers={}, cookies={}),
            {"user_id": "intruder"},
        )
    assert exc.value.status_code == 403

    response = await agent_routes.delete_session(
        "ended",
        SimpleNamespace(headers={}, cookies={}),
        {"user_id": "owner"},
    )

    assert response == {"status": "deleted", "session_id": "ended"}
    assert await store.load_session("ended") is None
    deleted = await store.load_session("ended", include_deleted=True)
    assert deleted is not None
    assert deleted["metadata"]["visibility"] == "deleted"


@pytest.mark.asyncio
async def test_check_session_access_returns_403_for_persisted_access_denied(
    tmp_path, monkeypatch
):
    store = LocalSessionStore(tmp_path)
    await store.init()
    await store.save_snapshot(
        session_id="persisted",
        user_id="owner",
        model="restored-model",
        messages=[{"role": "user", "content": "private"}],
    )

    manager = SessionManager()
    manager.persistence_store = store
    monkeypatch.setattr(agent_routes, "session_manager", manager)

    with pytest.raises(agent_routes.HTTPException) as exc:
        await agent_routes._check_session_access(
            "persisted",
            {"user_id": "intruder"},
            SimpleNamespace(headers={}, cookies={}),
        )

    assert exc.value.status_code == 403
    assert "persisted" not in manager.sessions


@pytest.mark.asyncio
async def test_session_access_passes_current_hf_token_when_restoring(
    tmp_path, monkeypatch
):
    store = LocalSessionStore(tmp_path)
    await store.init()
    await store.save_snapshot(
        session_id="persisted",
        user_id="owner",
        model="restored-model",
        messages=[{"role": "user", "content": "from disk"}],
    )

    manager = SessionManager()
    manager.persistence_store = store

    seen_tokens = []

    def fake_create_session_sync(**kwargs):
        seen_tokens.append(kwargs["hf_token"])
        session = _FakeSession(
            session_id=kwargs["session_id"],
            model=kwargs["model"],
        )
        return _FakeToolRouter(), session

    async def fake_run_session(*_args, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_create_session_sync", fake_create_session_sync)
    monkeypatch.setattr(manager, "_run_session", fake_run_session)
    monkeypatch.setattr(agent_routes, "session_manager", manager)

    agent_session = await agent_routes._check_session_access(
        "persisted",
        {"user_id": "owner"},
        SimpleNamespace(
            headers={"Authorization": "Bearer fresh-token"},
            cookies={},
        ),
    )

    assert seen_tokens == ["fresh-token"]
    assert agent_session.hf_token == "fresh-token"
    assert agent_session.session.hf_token == "fresh-token"

    agent_session.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await agent_session.task


@pytest.mark.asyncio
async def test_session_manager_restores_from_store_and_lists_runtime_session(
    tmp_path, monkeypatch
):
    store = LocalSessionStore(tmp_path)
    await store.init()
    await store.save_snapshot(
        session_id="persisted",
        user_id="owner",
        model="restored-model",
        messages=[{"role": "user", "content": "from disk"}],
        turn_count=3,
    )

    manager = SessionManager()
    manager.persistence_store = store

    def fake_create_session_sync(**kwargs):
        return _FakeToolRouter(), _FakeSession(
            session_id=kwargs["session_id"],
            model=kwargs["model"],
        )

    async def fake_run_session(*_args, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_create_session_sync", fake_create_session_sync)
    monkeypatch.setattr(manager, "_run_session", fake_run_session)
    monkeypatch.setattr(manager, "_cleanup_sandbox", lambda _session: None)

    agent_session = await manager.ensure_session_loaded("persisted", "owner")

    assert agent_session is not None
    assert agent_session.session.config.model_name == "restored-model"
    assert agent_session.session.turn_count == 3
    assert [msg.content for msg in agent_session.session.context_manager.items] == [
        "fresh system prompt",
        "from disk",
    ]
    listed = await manager.list_sessions("owner")
    assert listed[0]["session_id"] == "persisted"
    assert listed[0]["message_count"] == 2

    agent_session.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await agent_session.task


def test_last_event_seq_accepts_header_and_query_param():
    request = SimpleNamespace(
        headers={"last-event-id": "7"}, query_params={"after": "3"}
    )
    assert agent_routes._last_event_seq(request) == 7

    request = SimpleNamespace(headers={}, query_params={"after": "3"})
    assert agent_routes._last_event_seq(request) == 3

    request = SimpleNamespace(headers={"last-event-id": "not-an-int"}, query_params={})
    assert agent_routes._last_event_seq(request) == 0

    request = SimpleNamespace(headers={}, query_params={})
    assert agent_routes._last_event_seq(request) is None


@pytest.mark.asyncio
async def test_subscribe_events_without_replay_cursor_does_not_load_old_events(
    monkeypatch,
):
    class FailOnReplayStore:
        async def load_events_after(self, *_args, **_kwargs):
            raise AssertionError("default reconnect must not replay old events")

    class FakeBroadcaster:
        def subscribe(self):
            return 1, asyncio.Queue()

        def unsubscribe(self, _sub_id):
            pass

    class FakeManager:
        def _store(self):
            return FailOnReplayStore()

        async def ensure_session_loaded(self, *_args, **_kwargs):
            return SimpleNamespace(
                is_active=True,
                user_id="owner",
                broadcaster=FakeBroadcaster(),
            )

    monkeypatch.setattr(agent_routes, "session_manager", FakeManager())

    response = await agent_routes.subscribe_events(
        "persisted",
        SimpleNamespace(headers={}, query_params={}, cookies={}),
        {"user_id": "owner"},
    )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_sse_response_replays_events_with_ids_and_stops_on_terminal():
    unsubscribed = []
    broadcaster = SimpleNamespace(
        unsubscribe=lambda sub_id: unsubscribed.append(sub_id)
    )

    response = agent_routes._sse_response(
        broadcaster,
        asyncio.Queue(),
        42,
        replay_events=[
            {"seq": 1, "event_type": "processing", "data": {"message": "old"}},
            {"seq": 2, "event_type": "turn_complete", "data": {}},
        ],
        after_seq=1,
    )

    chunks = [chunk async for chunk in response.body_iterator]

    assert chunks == [
        'id: 2\ndata: {"event_type": "turn_complete", "data": {}, "seq": 2}\n\n'
    ]
    assert unsubscribed == [42]


@pytest.mark.asyncio
async def test_sse_response_deduplicates_live_events_after_replay():
    event_queue = asyncio.Queue()
    await event_queue.put({"seq": 2, "event_type": "processing", "data": {}})
    await event_queue.put({"seq": 3, "event_type": "turn_complete", "data": {}})

    response = agent_routes._sse_response(
        SimpleNamespace(unsubscribe=lambda _sub_id: None),
        event_queue,
        1,
        replay_events=[{"seq": 2, "event_type": "processing", "data": {}}],
        after_seq=1,
    )

    chunks = [chunk async for chunk in response.body_iterator]

    assert chunks == [
        'id: 2\ndata: {"event_type": "processing", "data": {}, "seq": 2}\n\n',
        'id: 3\ndata: {"event_type": "turn_complete", "data": {}, "seq": 3}\n\n',
    ]


@pytest.mark.asyncio
async def test_truncate_persists_snapshot_after_success(tmp_path):
    store = LocalSessionStore(tmp_path)
    await store.init()

    class TruncatingContext:
        def __init__(self):
            self.items = [
                Message(role="system", content="system"),
                Message(role="user", content="keep"),
                Message(role="assistant", content="drop"),
            ]

        def truncate_to_user_message(self, user_message_index):
            assert user_message_index == 0
            self.items = self.items[:2]
            return True

    session = _FakeSession(session_id="persisted", model="test-model")
    session.context_manager = TruncatingContext()
    manager = SessionManager()
    manager.persistence_store = store
    manager.sessions["persisted"] = AgentSession(
        session_id="persisted",
        session=session,
        tool_router=_FakeToolRouter(),
        submission_queue=asyncio.Queue(),
        user_id="owner",
    )

    assert await manager.truncate("persisted", 0) is True

    loaded = await store.load_session("persisted")
    assert loaded is not None
    assert loaded["metadata"]["message_count"] == 2
    assert loaded["messages"][-1]["role"] == "user"
    assert loaded["messages"][-1]["content"] == "keep"


@pytest.mark.asyncio
async def test_session_metadata_round_trips_through_snapshot_and_listing(tmp_path):
    store = LocalSessionStore(tmp_path)
    await store.init()

    manager = SessionManager()
    manager.persistence_store = store
    agent_session = AgentSession(
        session_id="phase8",
        session=_FakeSession(session_id="phase8", model="test-model"),
        tool_router=_FakeToolRouter(),
        submission_queue=asyncio.Queue(),
        user_id="owner",
        domain_id="call-center",
        provider_id="hf-jobs",
        dataset_repo="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    )
    manager.sessions["phase8"] = agent_session

    await manager.persist_session_snapshot(agent_session, runtime_state="idle")

    loaded = await store.load_session("phase8")
    assert loaded is not None
    assert loaded["metadata"]["domain_id"] == "call-center"
    assert loaded["metadata"]["provider_id"] == "hf-jobs"
    assert (
        loaded["metadata"]["dataset_repo"]
        == "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    )

    info = manager.get_session_info("phase8")
    assert info is not None
    assert info["domain_id"] == "call-center"
    assert info["provider_id"] == "hf-jobs"
    assert (
        info["dataset_repo"]
        == "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    )
