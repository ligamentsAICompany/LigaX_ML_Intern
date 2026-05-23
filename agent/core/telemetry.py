"""Best-effort telemetry helpers for agent and platform flows."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from agent.core.provenance import build_training_job_metadata
from agent.core.redact import scrub, scrub_string

try:
    from litellm import completion_cost
except Exception:  # pragma: no cover - defensive import fallback
    completion_cost = None

logger = logging.getLogger(__name__)
_heartbeat_tasks: set[asyncio.Task[Any]] = set()
_PUSH_TO_HUB_ASSIGNMENT_RE = re.compile(
    r"(?:^|[\s,(])push_to_hub\s*=\s*(true|false)\b", re.IGNORECASE
)


def extract_usage(response_or_chunk: Any) -> dict[str, int]:
    """Normalize LiteLLM usage into a flat integer payload."""
    usage = getattr(response_or_chunk, "usage", None)
    if usage is None and isinstance(response_or_chunk, dict):
        usage = response_or_chunk.get("usage")
    if usage is None:
        return {}

    def get(name: str, default: Any = 0) -> Any:
        if isinstance(usage, dict):
            return usage.get(name, default) or default
        return getattr(usage, name, default) or default

    prompt = get("prompt_tokens")
    completion = get("completion_tokens")
    total = get("total_tokens") or (prompt + completion)
    cache_read = get("cache_read_input_tokens")
    cache_creation = get("cache_creation_input_tokens")

    if not cache_read:
        details = get("prompt_tokens_details", None)
        if details is not None:
            if isinstance(details, dict):
                cache_read = details.get("cached_tokens", 0) or 0
            else:
                cache_read = getattr(details, "cached_tokens", 0) or 0

    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
        "cache_read_tokens": int(cache_read),
        "cache_creation_tokens": int(cache_creation),
    }


async def _emit(session: Any, event_type: str, data: dict[str, Any]) -> None:
    """Send a scrubbed telemetry event without raising into the caller."""
    if session is None or not hasattr(session, "send_event"):
        return
    try:
        from agent.core.session import Event

        await session.send_event(Event(event_type=event_type, data=scrub(data)))
    except Exception as exc:
        logger.debug("telemetry %s failed (non-fatal): %s", event_type, exc)


async def record_llm_call(
    session: Any,
    *,
    model: str,
    response: Any = None,
    latency_ms: int,
    finish_reason: str | None,
    kind: str = "main",
) -> dict[str, int]:
    """Emit an ``llm_call`` event and return extracted usage."""
    usage = extract_usage(response) if response is not None else {}
    cost_usd = 0.0
    if response is not None and completion_cost is not None:
        try:
            cost_usd = float(completion_cost(completion_response=response) or 0.0)
        except Exception:
            cost_usd = 0.0

    await _emit(
        session,
        "llm_call",
        {
            "model": model,
            "latency_ms": int(latency_ms),
            "finish_reason": finish_reason,
            "cost_usd": cost_usd,
            "kind": kind,
            **usage,
        },
    )
    return usage


def _infer_push_to_hub(script_or_cmd: Any) -> bool:
    if not isinstance(script_or_cmd, str):
        if isinstance(script_or_cmd, list):
            script_or_cmd = " ".join(str(part) for part in script_or_cmd)
        else:
            return False
    match = _PUSH_TO_HUB_ASSIGNMENT_RE.search(script_or_cmd)
    if match is None:
        return False
    return match.group(1).lower() == "true"


async def record_hf_job_submit(
    session: Any,
    job: Any,
    args: dict[str, Any],
    *,
    image: str,
    job_type: str,
) -> float:
    """Emit ``hf_job_submit`` and return the monotonic submit timestamp."""
    submit_ts = time.monotonic()
    script_text = args.get("script") or args.get("command") or ""
    job_metadata = build_training_job_metadata(args)
    await _emit(
        session,
        "hf_job_submit",
        {
            "job_id": getattr(job, "id", None),
            "job_url": getattr(job, "url", None),
            "flavor": args.get("hardware_flavor", "cpu-basic"),
            "timeout": args.get("timeout", "30m"),
            "job_type": job_type,
            "image": image,
            "namespace": args.get("namespace"),
            "push_to_hub": _infer_push_to_hub(script_text),
            **job_metadata,
        },
    )
    return submit_ts


async def record_hf_job_complete(
    session: Any,
    job: Any,
    *,
    flavor: str,
    final_status: str,
    submit_ts: float,
) -> None:
    await _emit(
        session,
        "hf_job_complete",
        {
            "job_id": getattr(job, "id", None),
            "flavor": flavor,
            "final_status": final_status,
            "wall_time_s": int(max(0, time.monotonic() - submit_ts)),
        },
    )


async def record_feedback(
    session: Any,
    *,
    rating: str,
    turn_index: int | None = None,
    message_id: str | None = None,
    comment: str | None = None,
) -> None:
    await _emit(
        session,
        "feedback",
        {
            "rating": rating,
            "turn_index": turn_index,
            "message_id": message_id,
            "comment": scrub_string(comment or "")[:500],
        },
    )


async def record_dataset_upload(
    session: Any,
    *,
    repo_id: str,
    filename: str,
    size_bytes: int,
    status: str,
) -> None:
    await _emit(
        session,
        "dataset_upload",
        {
            "repo_id": repo_id,
            "filename": filename,
            "size_bytes": int(size_bytes),
            "status": status,
        },
    )


def record_dataset_upload_sync(
    *,
    repo_id: str,
    filename: str,
    size_bytes: int,
    status: str,
) -> None:
    """Synchronous best-effort platform telemetry for routes without Session."""
    try:
        logger.info(
            "dataset_upload telemetry: %s",
            scrub(
                {
                    "repo_id": repo_id,
                    "filename": filename,
                    "size_bytes": int(size_bytes),
                    "status": status,
                }
            ),
        )
    except Exception as exc:
        logger.debug("dataset_upload telemetry failed (non-fatal): %s", exc)


async def record_model_chat(
    session: Any,
    *,
    model_id: str,
    message_count: int,
    status: str,
) -> None:
    await _emit(
        session,
        "model_chat",
        {
            "model_id": model_id,
            "message_count": int(message_count),
            "status": status,
        },
    )


def record_model_chat_sync(
    *,
    model_id: str,
    message_count: int,
    status: str,
) -> None:
    try:
        logger.info(
            "model_chat telemetry: %s",
            scrub(
                {
                    "model_id": model_id,
                    "message_count": int(message_count),
                    "status": status,
                }
            ),
        )
    except Exception as exc:
        logger.debug("model_chat telemetry failed (non-fatal): %s", exc)


async def record_provider_error(
    session: Any,
    *,
    provider: str,
    operation: str,
    error: Exception | str,
) -> None:
    await _emit(
        session,
        "provider_error",
        {
            "provider": provider,
            "operation": operation,
            "error": scrub_string(str(error))[:500],
        },
    )


class HeartbeatSaver:
    """Time-gated mid-turn session save helper."""

    @staticmethod
    def maybe_fire(session: Any) -> None:
        try:
            config = getattr(session, "config", None)
            if not getattr(config, "save_sessions", False):
                return
            interval = getattr(config, "heartbeat_interval_s", 0) or 0
            if interval <= 0:
                return

            now = time.monotonic()
            last = getattr(session, "_last_heartbeat_ts", None)
            if last is None:
                session._last_heartbeat_ts = now
                return
            if now - last < interval:
                return

            session._last_heartbeat_ts = now
            repo_id = getattr(config, "session_dataset_repo", None)
            if not repo_id:
                return

            try:
                task = asyncio.get_running_loop().create_task(
                    asyncio.to_thread(session.save_and_upload_detached, repo_id)
                )
                _heartbeat_tasks.add(task)
                task.add_done_callback(_heartbeat_tasks.discard)
            except RuntimeError:
                try:
                    session.save_and_upload_detached(repo_id)
                except Exception as exc:
                    logger.debug("heartbeat save failed (non-fatal): %s", exc)
        except Exception as exc:
            logger.debug("heartbeat scheduling failed (non-fatal): %s", exc)
