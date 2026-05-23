"""Optional durable session persistence for backend sessions.

The backend must run in local dev and tests without an external database. This
module exposes one small async store interface plus a no-op implementation and
a JSON-file local store used as the default fallback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.core.redact import scrub

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_STORE_DIR = _PROJECT_ROOT / ".session_store"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    return value


def _safe_session_filename(session_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)
    return f"{safe or 'session'}.json"


class NoopSessionStore:
    """Async no-op store used when persistence is explicitly disabled."""

    enabled = False

    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert_session(self, **_: Any) -> None:
        return None

    async def save_snapshot(self, **_: Any) -> None:
        return None

    async def load_session(self, *_: Any, **__: Any) -> dict[str, Any] | None:
        return None

    async def list_sessions(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    async def soft_delete_session(self, *_: Any, **__: Any) -> None:
        return None

    async def update_session_fields(self, *_: Any, **__: Any) -> None:
        return None

    async def append_event(self, *_: Any, **__: Any) -> int | None:
        return None

    async def load_events_after(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []


class LocalSessionStore(NoopSessionStore):
    """JSON-file session store for local/dev durable fallback."""

    enabled = True

    def __init__(self, root_dir: str | os.PathLike[str] | None = None) -> None:
        self.root_dir = Path(root_dir or _DEFAULT_STORE_DIR)
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            self._records = {}
            for path in self.root_dir.glob("*.json"):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning(
                        "Skipping unreadable session store file %s: %s", path, e
                    )
                    continue
                session_id = record.get("metadata", {}).get("session_id") or record.get(
                    "session_id"
                )
                if isinstance(session_id, str) and session_id:
                    self._records[session_id] = record

    async def close(self) -> None:
        return None

    def _path_for(self, session_id: str) -> Path:
        return self.root_dir / _safe_session_filename(session_id)

    def _write_record(self, session_id: str) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(session_id)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._records[session_id], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    async def upsert_session(
        self,
        *,
        session_id: str,
        user_id: str,
        model: str,
        created_at: datetime | str | None = None,
        runtime_state: str = "idle",
        status: str = "active",
        message_count: int = 0,
        turn_count: int = 0,
        pending_approval: list[dict[str, Any]] | None = None,
        claude_counted: bool = False,
        **extra_fields: Any,
    ) -> None:
        async with self._lock:
            now = _now_iso()
            record = self._records.setdefault(
                session_id,
                {"schema_version": SCHEMA_VERSION, "messages": [], "events": []},
            )
            metadata = record.setdefault("metadata", {})
            metadata.setdefault("created_at", _jsonable(created_at) or now)
            metadata.update(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "model": model,
                    "runtime_state": runtime_state,
                    "status": status,
                    "message_count": int(message_count),
                    "turn_count": int(turn_count),
                    "pending_approval": scrub(_jsonable(pending_approval or [])),
                    "claude_counted": bool(claude_counted),
                    "updated_at": now,
                    "schema_version": SCHEMA_VERSION,
                    "visibility": metadata.get("visibility", "live"),
                }
            )
            metadata.update(scrub(_jsonable(extra_fields)))
            self._write_record(session_id)

    async def save_snapshot(
        self,
        *,
        session_id: str,
        user_id: str,
        model: str,
        messages: list[dict[str, Any]],
        runtime_state: str = "idle",
        status: str = "active",
        turn_count: int = 0,
        pending_approval: list[dict[str, Any]] | None = None,
        claude_counted: bool = False,
        created_at: datetime | str | None = None,
        **extra_fields: Any,
    ) -> None:
        await self.upsert_session(
            session_id=session_id,
            user_id=user_id,
            model=model,
            created_at=created_at,
            runtime_state=runtime_state,
            status=status,
            message_count=len(messages),
            turn_count=turn_count,
            pending_approval=pending_approval,
            claude_counted=claude_counted,
            **extra_fields,
        )
        async with self._lock:
            self._records[session_id]["messages"] = scrub(_jsonable(messages))
            self._write_record(session_id)

    async def load_session(
        self, session_id: str, *, include_deleted: bool = False
    ) -> dict[str, Any] | None:
        async with self._lock:
            record = self._records.get(session_id)
            if not record:
                return None
            metadata = dict(record.get("metadata") or {})
            if metadata.get("visibility") == "deleted" and not include_deleted:
                return None
            return {
                "metadata": metadata,
                "messages": list(record.get("messages") or []),
            }

    async def list_sessions(
        self, user_id: str, *, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        async with self._lock:
            rows: list[dict[str, Any]] = []
            for record in self._records.values():
                metadata = dict(record.get("metadata") or {})
                if not include_deleted and metadata.get("visibility") == "deleted":
                    continue
                if user_id != "dev" and metadata.get("user_id") != user_id:
                    continue
                rows.append(metadata)
            rows.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
            return rows

    async def soft_delete_session(self, session_id: str) -> None:
        async with self._lock:
            record = self._records.get(session_id)
            if not record:
                return
            metadata = record.setdefault("metadata", {})
            metadata["visibility"] = "deleted"
            metadata["runtime_state"] = "idle"
            metadata["updated_at"] = _now_iso()
            self._write_record(session_id)

    async def update_session_fields(self, session_id: str, **fields: Any) -> None:
        if not fields:
            return
        async with self._lock:
            record = self._records.get(session_id)
            if not record:
                return
            metadata = record.setdefault("metadata", {})
            metadata.update(scrub(_jsonable(fields)))
            metadata["updated_at"] = _now_iso()
            self._write_record(session_id)

    async def append_event(
        self, session_id: str, event_type: str, data: dict[str, Any] | None
    ) -> int | None:
        async with self._lock:
            record = self._records.setdefault(
                session_id,
                {
                    "schema_version": SCHEMA_VERSION,
                    "metadata": {
                        "session_id": session_id,
                        "user_id": "dev",
                        "model": None,
                        "created_at": _now_iso(),
                        "updated_at": _now_iso(),
                        "visibility": "live",
                        "message_count": 0,
                        "turn_count": 0,
                    },
                    "messages": [],
                    "events": [],
                },
            )
            events = record.setdefault("events", [])
            seq = int(events[-1]["seq"]) + 1 if events else 1
            events.append(
                {
                    "seq": seq,
                    "event_type": event_type,
                    "data": scrub(_jsonable(data or {})),
                }
            )
            record.setdefault("metadata", {})["updated_at"] = _now_iso()
            self._write_record(session_id)
            return seq

    async def load_events_after(
        self, session_id: str, after_seq: int = 0
    ) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._records.get(session_id)
            if not record:
                return []
            after = int(after_seq or 0)
            return [
                dict(event)
                for event in record.get("events", [])
                if int(event.get("seq") or 0) > after
            ]


_store: NoopSessionStore | LocalSessionStore | None = None


def get_session_store() -> NoopSessionStore | LocalSessionStore:
    global _store
    if _store is None:
        disabled = os.environ.get("SESSION_PERSISTENCE_DISABLED", "").lower()
        if disabled in {"1", "true", "yes"}:
            _store = NoopSessionStore()
        else:
            _store = LocalSessionStore(os.environ.get("SESSION_PERSISTENCE_DIR"))
    return _store


def _reset_store_for_tests(
    store: NoopSessionStore | LocalSessionStore | None = None,
) -> None:
    global _store
    _store = store
