"""Phase 4 tests for trace redaction and telemetry safety."""

import json
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from litellm import Message

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core.session import Session  # noqa: E402
from agent.core.session_persistence import LocalSessionStore  # noqa: E402
from agent.core.telemetry import (  # noqa: E402
    HeartbeatSaver,
    record_dataset_upload,
    record_hf_job_complete,
    record_hf_job_submit,
    record_llm_call,
    record_model_chat,
    record_provider_error,
)
from agent.core.redact import scrub, scrub_string  # noqa: E402
from services import dataset_service, model_chat_service  # noqa: E402


def test_scrub_string_redacts_provider_tokens_and_indian_pii():
    secret_text = (
        "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890 "
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456 "
        "OpenAI sk-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN1234567890 "
        "github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_extra "
        "AWS AKIAABCDEFGHIJKLMNOP "
        "Aadhaar 1234 5678 9012 "
        "PAN ABCDE1234F "
        "GSTIN 27ABCDE1234F1Z5 "
        "IFSC HDFC0001234 "
        "email rahul@example.com phone +91 98765 43210"
    )

    scrubbed = scrub_string(secret_text)

    assert "hf_abcdefghijklmnopqrstuvwxyz1234567890" not in scrubbed
    assert "Bearer abcdefghijklmnopqrstuvwxyz123456" not in scrubbed
    assert "sk-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN1234567890" not in scrubbed
    assert "github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_extra" not in scrubbed
    assert "AKIAABCDEFGHIJKLMNOP" not in scrubbed
    assert "1234 5678 9012" not in scrubbed
    assert "ABCDE1234F" not in scrubbed
    assert "27ABCDE1234F1Z5" not in scrubbed
    assert "HDFC0001234" not in scrubbed
    assert "rahul@example.com" not in scrubbed
    assert "98765 43210" not in scrubbed
    assert "[REDACTED_HF_TOKEN]" in scrubbed
    assert "[REDACTED_AADHAAR]" in scrubbed


def test_scrub_recursively_redacts_secret_like_keys_without_mutating_input():
    payload = {
        "messages": [
            {
                "content": "invoice INV-2026-0001 for customer Priya at 12 MG Road",
                "metadata": {"password": "plain-text-secret"},
            }
        ],
        "env": {
            "OPENAI_API_KEY": "sk-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN1234567890"
        },
        "tuple": ("phone 9876543210",),
    }

    scrubbed = scrub(payload)

    assert scrubbed is not payload
    assert scrubbed["messages"][0]["metadata"]["password"] == "[REDACTED_SECRET]"
    assert scrubbed["env"]["OPENAI_API_KEY"] == "[REDACTED_SECRET]"
    assert "INV-2026-0001" not in scrubbed["messages"][0]["content"]
    assert "Priya" not in scrubbed["messages"][0]["content"]
    assert "12 MG Road" not in scrubbed["messages"][0]["content"]
    assert "9876543210" not in scrubbed["tuple"][0]
    assert payload["messages"][0]["metadata"]["password"] == "plain-text-secret"


def test_scrub_preserves_token_usage_metrics_but_redacts_secret_keys():
    payload = {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "cache_read_tokens": 3,
        "cache_creation_tokens": 2,
        "HF_TOKEN": "hf_abcdefghijklmnopqrstuvwxyz1234567890",
        "Authorization token": "Bearer abcdefghijklmnopqrstuvwxyz123456",
        "api_key": "sk-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN1234567890",
        "access_token": "secret-access-token",
        "refresh_token": "secret-refresh-token",
        "bearer": "secret-bearer-token",
        "password": "secret-password",
        "secret": "secret-value",
    }

    scrubbed = scrub(payload)

    assert scrubbed["prompt_tokens"] == 11
    assert scrubbed["completion_tokens"] == 7
    assert scrubbed["total_tokens"] == 18
    assert scrubbed["cache_read_tokens"] == 3
    assert scrubbed["cache_creation_tokens"] == 2
    for key in (
        "HF_TOKEN",
        "Authorization token",
        "api_key",
        "access_token",
        "refresh_token",
        "bearer",
        "password",
        "secret",
    ):
        assert scrubbed[key] == "[REDACTED_SECRET]"


def test_scrub_handles_cyclic_and_deep_payloads_without_crashing():
    cyclic = {"token_count": 5}
    cyclic["self"] = cyclic

    scrubbed_cycle = scrub(cyclic)

    assert scrubbed_cycle["token_count"] == 5
    assert scrubbed_cycle["self"] == "[REDACTED_CYCLE]"

    deep: object = "safe leaf"
    for _ in range(80):
        deep = {"nested": deep}

    scrubbed_deep = scrub(deep)

    cursor = scrubbed_deep
    for _ in range(50):
        assert isinstance(cursor, dict)
        cursor = cursor["nested"]
    assert cursor == "[REDACTED_MAX_DEPTH]"


def test_save_trajectory_local_redacts_messages_and_events(tmp_path):
    session = Session.__new__(Session)
    session.session_id = "phase4"
    session.session_start_time = "2026-05-20T10:00:00"
    session.config = SimpleNamespace(model_name="test-model")
    session.context_manager = SimpleNamespace(
        items=[Message(role="user", content="my PAN is ABCDE1234F")]
    )
    session.logged_events = [
        {
            "timestamp": "2026-05-20T10:00:01",
            "event_type": "tool_log",
            "data": {"log": "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890"},
        }
    ]

    path = session.save_trajectory_local(directory=str(tmp_path))

    assert path is not None
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    serialized = json.dumps(saved)
    assert "ABCDE1234F" not in serialized
    assert "hf_abcdefghijklmnopqrstuvwxyz1234567890" not in serialized
    assert "[REDACTED_PAN]" in serialized
    assert "[REDACTED_HF_TOKEN]" in serialized


def test_session_uploader_creates_private_dataset_by_default(tmp_path, monkeypatch):
    created_repos = []

    class FakeApi:
        def create_repo(self, **kwargs):
            created_repos.append(kwargs)

        def upload_file(self, **_kwargs):
            return None

    session_file = tmp_path / "session.json"
    session_file.write_text(
        json.dumps(
            {
                "session_id": "phase4",
                "session_start_time": "2026-05-20T10:00:00",
                "session_end_time": "2026-05-20T10:01:00",
                "model_name": "test-model",
                "messages": [],
                "events": [],
                "upload_status": "pending",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("HF_SESSION_UPLOAD_TOKEN", "hf_test_session_upload_token")

    from agent.core import session_uploader

    session_uploader = importlib.reload(session_uploader)
    monkeypatch.setattr(session_uploader, "HfApi", FakeApi, raising=False)
    monkeypatch.setattr(
        session_uploader, "_SESSION_TOKEN", "hf_test_session_upload_token"
    )

    assert session_uploader.upload_session_as_file(str(session_file), "tester/sessions")
    assert created_repos[0]["private"] is True


@pytest.mark.asyncio
async def test_local_session_store_redacts_snapshots_and_events(tmp_path):
    store = LocalSessionStore(tmp_path)
    await store.init()

    await store.save_snapshot(
        session_id="persisted",
        user_id="owner",
        model="test-model",
        messages=[{"role": "user", "content": "email me at user@example.com"}],
        pending_approval=[
            {
                "tool": "hf_jobs",
                "tool_call_id": "tc1",
                "arguments": {
                    "script": "print('Aadhaar 1234 5678 9012')",
                    "secrets": {"HF_TOKEN": "hf_abcdefghijklmnopqrstuvwxyz1234567890"},
                },
            }
        ],
    )
    await store.append_event(
        "persisted", "tool_output", {"output": "GSTIN 27ABCDE1234F1Z5"}
    )

    loaded = await store.load_session("persisted")
    assert loaded is not None
    events = await store.load_events_after("persisted", 0)
    serialized = json.dumps({"loaded": loaded, "events": events})
    assert "user@example.com" not in serialized
    assert "1234 5678 9012" not in serialized
    assert "hf_abcdefghijklmnopqrstuvwxyz1234567890" not in serialized
    assert "27ABCDE1234F1Z5" not in serialized


class _FakeSession:
    def __init__(self, *, raise_on_send: bool = False):
        self.events = []
        self.raise_on_send = raise_on_send
        self.config = SimpleNamespace(
            save_sessions=False,
            heartbeat_interval_s=0,
            session_dataset_repo="tester/sessions",
        )

    async def send_event(self, event):
        if self.raise_on_send:
            raise RuntimeError("telemetry sink unavailable")
        self.events.append(event)


@pytest.mark.asyncio
async def test_telemetry_records_redacted_event_shapes_and_never_raises(monkeypatch):
    session = _FakeSession()
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )
    monkeypatch.setattr(
        "agent.core.telemetry.completion_cost",
        lambda completion_response: 0.0123,
    )

    usage = await record_llm_call(
        session,
        model="test-model",
        response=response,
        latency_ms=123,
        finish_reason="stop",
    )
    submit_ts = await record_hf_job_submit(
        session,
        SimpleNamespace(id="job1", url="https://hf.co/jobs/job1"),
        {
            "hardware_flavor": "cpu-basic",
            "timeout": "30m",
            "script": "print('hf_abcdefghijklmnopqrstuvwxyz1234567890')",
            "namespace": "tester",
        },
        image="python:3.12",
        job_type="Python",
    )
    await record_hf_job_complete(
        session,
        SimpleNamespace(id="job1"),
        flavor="cpu-basic",
        final_status="COMPLETED",
        submit_ts=submit_ts,
    )
    await record_dataset_upload(
        session,
        repo_id="tester/private-dataset",
        filename="train.csv",
        size_bytes=128,
        status="success",
    )
    await record_model_chat(
        session,
        model_id="tester/model",
        message_count=2,
        status="started",
    )
    await record_provider_error(
        _FakeSession(raise_on_send=True),
        provider="hf",
        operation="model_chat",
        error=RuntimeError("Bearer abcdefghijklmnopqrstuvwxyz123456 failed"),
    )

    assert usage["total_tokens"] == 7
    event_types = [event.event_type for event in session.events]
    assert event_types == [
        "llm_call",
        "hf_job_submit",
        "hf_job_complete",
        "dataset_upload",
        "model_chat",
    ]
    serialized = json.dumps([event.data for event in session.events])
    assert "hf_abcdefghijklmnopqrstuvwxyz1234567890" not in serialized
    assert session.events[0].data["cost_usd"] == 0.0123
    assert session.events[1].data["push_to_hub"] is False


def test_heartbeat_saver_is_best_effort():
    session = _FakeSession()
    session.config.save_sessions = True
    session.config.heartbeat_interval_s = 1
    session.save_and_upload_detached = lambda _repo: (_ for _ in ()).throw(
        RuntimeError("disk unavailable")
    )

    HeartbeatSaver.maybe_fire(session)
    session._last_heartbeat_ts -= 2
    HeartbeatSaver.maybe_fire(session)


def test_dataset_upload_service_records_telemetry(monkeypatch):
    captured = []

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            return {"name": "tester"}

        def create_repo(self, **_kwargs):
            return None

        def upload_file(self, **_kwargs):
            return None

    monkeypatch.setattr(dataset_service, "HfApi", FakeApi)
    monkeypatch.setattr(
        dataset_service.telemetry,
        "record_dataset_upload_sync",
        lambda **kwargs: captured.append(kwargs),
    )

    payload = dataset_service.upload_dataset_contents(
        safe_filename="train.csv",
        contents=b"a,b\n1,2\n",
        repo_id="tester/dataset",
        token="valid",
    )

    assert payload["dataset_id"] == "tester/dataset"
    assert captured == [
        {
            "repo_id": "tester/dataset",
            "filename": "train.csv",
            "size_bytes": 8,
            "status": "success",
        }
    ]


@pytest.mark.asyncio
async def test_model_chat_service_records_telemetry(monkeypatch):
    captured = []

    class FakeStreamResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_text(self):
            yield "data: hello\n\n"

        async def aread(self):
            return b""

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, headers, json):
            return FakeStreamResponse()

    monkeypatch.setattr(
        model_chat_service.telemetry,
        "record_model_chat_sync",
        lambda **kwargs: captured.append(kwargs),
    )

    chunks = [
        chunk
        async for chunk in model_chat_service.stream_model_chat(
            token="valid",
            model_id="tester/model",
            messages=[{"role": "user", "content": "hi"}],
            async_client_factory=FakeAsyncClient,
        )
    ]

    assert chunks == ["data: hello\n\n"]
    assert captured == [
        {"model_id": "tester/model", "message_count": 1, "status": "started"},
        {"model_id": "tester/model", "message_count": 1, "status": "success"},
    ]


@pytest.mark.asyncio
async def test_model_chat_sse_exception_is_scrubbed(monkeypatch):
    class FailingAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, headers, json):
            raise RuntimeError(
                "failed with HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890"
            )

    monkeypatch.setattr(
        model_chat_service.telemetry,
        "record_model_chat_sync",
        lambda **_kwargs: None,
    )

    chunks = [
        chunk
        async for chunk in model_chat_service.stream_model_chat(
            token="valid",
            model_id="tester/model",
            messages=[{"role": "user", "content": "hi"}],
            async_client_factory=FailingAsyncClient,
        )
    ]

    assert "hf_abcdefghijklmnopqrstuvwxyz1234567890" not in chunks[0]
    assert "[REDACTED_HF_TOKEN]" in chunks[0]


def test_dataset_upload_exception_details_are_scrubbed(monkeypatch):
    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            return {"name": "tester"}

        def create_repo(self, **_kwargs):
            raise RuntimeError(
                "create failed with HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890"
            )

    monkeypatch.setattr(dataset_service, "HfApi", FakeApi)

    with pytest.raises(HTTPException) as exc_info:
        dataset_service.upload_dataset_contents(
            safe_filename="train.csv",
            contents=b"a,b\n1,2\n",
            repo_id="tester/dataset",
            token="valid",
        )

    assert "hf_abcdefghijklmnopqrstuvwxyz1234567890" not in exc_info.value.detail
    assert "[REDACTED_HF_TOKEN]" in exc_info.value.detail
