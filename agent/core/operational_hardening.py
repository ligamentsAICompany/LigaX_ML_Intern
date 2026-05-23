"""Operational health, KPI, and orphan-job sweep helpers.

The functions in this module are shared by the read-only health route and the
admin CLI. Mutation-capable operations stay out of unauthenticated HTTP routes.
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from agent.core.cloud_providers import ProviderContext, get_cloud_provider_registry
from agent.core.redact import scrub

HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "INFERENCE_TOKEN")
_SAFE_ORPHAN_RUNTIME_STATES = {"ended", "idle"}
_ACTIVE_RUNTIME_STATES = {"processing", "waiting_approval"}


def _is_configured_env(name: str) -> bool:
    return bool(os.environ.get(name))


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def build_provider_health_snapshot(hf_token: str | None = None) -> dict[str, Any]:
    """Return provider readiness without exposing credentials or token values."""
    registry = get_cloud_provider_registry()
    token_sources = [name for name in HF_TOKEN_ENV_VARS if _is_configured_env(name)] + (
        ["request"] if hf_token else []
    )
    hf_token_configured = bool(token_sources)

    providers: dict[str, Any] = {}
    for provider in registry.all():
        credential_env_vars = tuple(getattr(provider, "credential_env_vars", ()) or ())
        configured = [name for name in credential_env_vars if _is_configured_env(name)]
        missing = [name for name in credential_env_vars if name not in configured]

        if provider.provider_id == "hf-jobs":
            ready = bool(
                provider.enabled and provider.executable and hf_token_configured
            )
            missing = [] if hf_token_configured else list(HF_TOKEN_ENV_VARS)
        else:
            ready = bool(provider.enabled and not provider.executable and not missing)

        providers[provider.provider_id] = {
            "display_name": provider.display_name,
            "enabled": bool(provider.enabled),
            "executable": bool(provider.executable),
            "ready": ready,
            "mode": "executable" if provider.executable else "plan-only",
            "configured": configured,
            "missing": missing,
            "disabled_reason": provider.disabled_reason,
        }

    status = "ok" if all(p["ready"] for p in providers.values()) else "degraded"
    return {
        "status": status,
        "hf_token": {
            "configured": hf_token_configured,
            "sources": token_sources,
        },
        "providers": providers,
    }


async def build_kpi_snapshot(
    store: Any,
    *,
    live_sessions: int | None = None,
) -> dict[str, Any]:
    """Build a compact operational KPI snapshot from the persistence store."""
    rows = await store.list_sessions("dev", include_deleted=True)
    provider_counts: Counter[str] = Counter()
    session_counts: Counter[str] = Counter()
    job_counts: Counter[str] = Counter()

    for row in rows:
        visibility = row.get("visibility", "live")
        runtime_state = row.get("runtime_state")
        status = row.get("status")
        provider_id = str(row.get("provider_id") or "unknown")
        running_job_ids = [
            job_id
            for job_id in row.get("running_job_ids", [])
            if isinstance(job_id, str)
        ]

        provider_counts[provider_id] += 1
        job_counts["tracked_running_job_ids"] += len(running_job_ids)
        if visibility == "deleted":
            session_counts["deleted"] += 1
        if status == "ended" or runtime_state == "ended":
            session_counts["ended"] += 1
        elif visibility != "deleted":
            session_counts["active"] += 1

    records = getattr(store, "_records", {}) or {}
    event_counts: Counter[str] = Counter()
    for record in records.values():
        for event in record.get("events", []) or []:
            event_type = str(event.get("event_type") or "unknown")
            event_counts["total"] += 1
            event_counts[event_type] += 1
            if event_type == "error":
                event_counts["errors"] += 1

    return {
        "sessions": {
            "active": int(session_counts["active"]),
            "ended": int(session_counts["ended"]),
            "deleted": int(session_counts["deleted"]),
            "live_runtime": live_sessions,
            "persisted_total": len(rows),
        },
        "providers": dict(provider_counts),
        "jobs": dict(job_counts),
        "events": dict(event_counts),
    }


def _safe_orphan_reason(row: dict[str, Any], *, stale_after: datetime) -> str | None:
    visibility = row.get("visibility", "live")
    runtime_state = str(row.get("runtime_state") or "idle")
    status = str(row.get("status") or "active")

    if visibility == "deleted":
        return "deleted_session"
    if status == "ended" or runtime_state == "ended":
        return "ended_session"
    if runtime_state in _ACTIVE_RUNTIME_STATES:
        return None

    updated_at = _parse_datetime(row.get("updated_at"))
    if (
        updated_at
        and updated_at < stale_after
        and runtime_state in _SAFE_ORPHAN_RUNTIME_STATES
    ):
        return "stale_idle_session"
    return None


async def sweep_orphan_jobs(
    store: Any,
    *,
    apply: bool = False,
    hf_token: str | None = None,
    stale_hours: int = 24,
) -> dict[str, Any]:
    """Dry-run or safely cancel tracked jobs from ended/deleted/stale sessions."""
    rows = await store.list_sessions("dev", include_deleted=True)
    stale_after = datetime.now(UTC) - timedelta(hours=stale_hours)
    candidates: list[dict[str, Any]] = []

    for row in rows:
        reason = _safe_orphan_reason(row, stale_after=stale_after)
        if not reason:
            continue
        session_id = str(row.get("session_id") or "")
        provider_id = str(row.get("provider_id") or "hf-jobs")
        for job_id in row.get("running_job_ids", []) or []:
            if not isinstance(job_id, str) or not job_id:
                continue
            candidates.append(
                {
                    "session_id": session_id,
                    "provider_id": provider_id,
                    "job_id": job_id,
                    "reason": reason,
                }
            )

    cancelled: list[str] = []
    errors: list[dict[str, str]] = []
    if apply:
        by_session: dict[tuple[str, str], list[str]] = {}
        for candidate in candidates:
            key = (candidate["session_id"], candidate["provider_id"])
            by_session.setdefault(key, []).append(candidate["job_id"])

        registry = get_cloud_provider_registry()
        for (session_id, provider_id), job_ids in by_session.items():
            try:
                provider = (
                    registry.require_executable(provider_id)
                    if provider_id == "hf-jobs"
                    else registry.require(provider_id)
                )
                await provider.cancel_jobs(
                    job_ids,
                    ProviderContext(
                        hf_token=hf_token,
                        tool_name="phase11_orphan_sweep",
                        provider_id=provider_id,
                    ),
                )
                await store.update_session_fields(session_id, running_job_ids=[])
                cancelled.extend(job_ids)
            except Exception as exc:
                errors.append(
                    {
                        "session_id": session_id,
                        "provider_id": provider_id,
                        "error": str(exc)[:500],
                    }
                )

    return scrub(
        {
            "applied": bool(apply),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "cancelled_job_ids": cancelled,
            "errors": errors,
        }
    )
