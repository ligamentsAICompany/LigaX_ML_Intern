"""Agent API routes — REST + SSE endpoints.

All routes (except /health) require authentication via the get_current_user
dependency. In dev mode (no OAUTH_CLIENT_ID), auth is bypassed automatically.
"""

import asyncio
import json
import logging
import os
from typing import Any

from dependencies import (
    _extract_token,
    get_current_user,
    require_huggingface_org_member,
)
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse
from litellm import acompletion
from models import (
    ApprovalRequest,
    HealthResponse,
    LLMHealthResponse,
    SessionInfo,
    SessionResponse,
    SubmitRequest,
    TruncateRequest,
)
from session_manager import (
    MAX_SESSIONS,
    AgentSession,
    SessionCapacityError,
    session_manager,
)

import user_quotas

from agent.core.auto_finetune import credential_readiness, resolve_auto_finetune_config
from agent.core.llm_params import _resolve_llm_params
from agent.core.operational_hardening import build_provider_health_snapshot
from domain_templates import (
    list_domain_templates,
    normalize_workflow_metadata,
    render_workflow_context,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agent"])

AVAILABLE_MODELS = [
    {
        "id": "moonshotai/Kimi-K2.6",
        "label": "Kimi K2.6",
        "provider": "huggingface",
        "tier": "free",
        "recommended": True,
    },
    {
        "id": "bedrock/us.anthropic.claude-opus-4-6-v1",
        "label": "Claude Opus 4.6",
        "provider": "anthropic",
        "tier": "pro",
        "recommended": True,
    },
    {
        "id": "MiniMaxAI/MiniMax-M2.7",
        "label": "MiniMax M2.7",
        "provider": "huggingface",
        "tier": "free",
    },
    {
        "id": "zai-org/GLM-5.1",
        "label": "GLM 5.1",
        "provider": "huggingface",
        "tier": "free",
    },
]


def _is_anthropic_model(model_id: str) -> bool:
    return "anthropic" in model_id


async def _require_hf_for_anthropic(request: Request, model_id: str) -> None:
    """403 if a non-``huggingface``-org user tries to select an Anthropic model.

    Anthropic models are billed to the Space's ``ANTHROPIC_API_KEY``; every
    other model in ``AVAILABLE_MODELS`` is routed through HF Router and
    billed via ``X-HF-Bill-To``. The gate only fires for Anthropic so
    non-HF users can still freely switch between the free models.

    Pattern: https://github.com/huggingface/ml-intern/pull/63
    """
    if not _is_anthropic_model(model_id):
        return
    if not await require_huggingface_org_member(request):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "anthropic_restricted",
                "message": (
                    "Opus is gated to HF staff. Pick a free model — "
                    "Kimi K2.6, MiniMax M2.7, or GLM 5.1 — instead."
                ),
            },
        )


async def _enforce_claude_quota(
    user: dict[str, Any],
    agent_session: AgentSession,
) -> None:
    """Charge the user's daily Claude quota on first use of Anthropic in a session.

    Runs at *message-submit* time, not session-create time — so spinning up a
    Claude session to look around doesn't burn quota. The ``claude_counted``
    flag on ``AgentSession`` guards against re-counting the same session.

    No-ops when the session's current model isn't Anthropic, or when this
    session has already been charged. Raises 429 when the user has hit
    their daily cap.
    """
    if agent_session.claude_counted:
        return
    model_name = agent_session.session.config.model_name
    if not _is_anthropic_model(model_name):
        return
    user_id = user["user_id"]
    used = await user_quotas.get_claude_used_today(user_id)
    cap = user_quotas.daily_cap_for(user.get("plan"))
    if used >= cap:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "claude_daily_cap",
                "plan": user.get("plan", "free"),
                "cap": cap,
                "message": (
                    "Daily Claude limit reached. Upgrade to HF Pro for "
                    f"{user_quotas.CLAUDE_PRO_DAILY}/day or use a free model."
                ),
            },
        )
    await user_quotas.increment_claude(user_id)
    agent_session.claude_counted = True


async def _check_session_access(
    session_id: str,
    user: dict[str, Any],
    request: Request | None = None,
) -> AgentSession:
    """Verify the user has access to the given session. Raises 403 or 404."""
    hf_token = _extract_token(request) if request is not None else None
    agent_session = await session_manager.ensure_session_loaded(
        session_id,
        user["user_id"],
        hf_token=hf_token,
    )
    if not agent_session:
        persisted = await session_manager._store().load_session(session_id)
        if persisted:
            owner = str((persisted.get("metadata") or {}).get("user_id") or "dev")
            if user["user_id"] != "dev" and owner not in {user["user_id"], "dev"}:
                raise HTTPException(
                    status_code=403, detail="Access denied to this session"
                )
        raise HTTPException(status_code=404, detail="Session not found")
    if user["user_id"] != "dev" and agent_session.user_id not in {
        user["user_id"],
        "dev",
    }:
        raise HTTPException(status_code=403, detail="Access denied to this session")
    return agent_session


async def _check_session_delete_access(session_id: str, user: dict[str, Any]) -> None:
    """Allow owners to delete active or ended sessions without restoring ended ones."""
    agent_session = session_manager.sessions.get(session_id)
    if agent_session:
        if user["user_id"] == "dev" or agent_session.user_id in {
            user["user_id"],
            "dev",
        }:
            return
        raise HTTPException(status_code=403, detail="Access denied to this session")

    persisted = await session_manager._store().load_session(session_id)
    if persisted:
        owner = str((persisted.get("metadata") or {}).get("user_id") or "dev")
        if user["user_id"] == "dev" or owner in {user["user_id"], "dev"}:
            return
        raise HTTPException(status_code=403, detail="Access denied to this session")

    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        active_sessions=session_manager.active_session_count,
        max_sessions=MAX_SESSIONS,
    )


@router.get("/health/providers")
async def provider_health_check(request: Request) -> dict[str, Any]:
    """Read-only provider readiness endpoint. Does not expose credentials."""
    hf_token = _extract_token(request) or os.environ.get("HF_TOKEN")
    return build_provider_health_snapshot(hf_token=hf_token)


@router.get("/health/llm", response_model=LLMHealthResponse)
async def llm_health_check() -> LLMHealthResponse:
    """Check if the LLM provider is reachable and the API key is valid.

    Makes a minimal 1-token completion call.  Catches common errors:
    - 401 → invalid API key
    - 402/insufficient_quota → out of credits
    - 429 → rate limited
    - timeout / network → provider unreachable
    """
    model = session_manager.config.model_name
    try:
        llm_params = _resolve_llm_params(model, reasoning_effort="high")
        await acompletion(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            timeout=10,
            **llm_params,
        )
        return LLMHealthResponse(status="ok", model=model)
    except Exception as e:
        err_str = str(e).lower()
        error_type = "unknown"

        if (
            "401" in err_str
            or "auth" in err_str
            or "invalid" in err_str
            or "api key" in err_str
        ):
            error_type = "auth"
        elif (
            "402" in err_str
            or "credit" in err_str
            or "quota" in err_str
            or "insufficient" in err_str
            or "billing" in err_str
        ):
            error_type = "credits"
        elif "429" in err_str or "rate" in err_str:
            error_type = "rate_limit"
        elif "timeout" in err_str or "connect" in err_str or "network" in err_str:
            error_type = "network"

        logger.warning(f"LLM health check failed ({error_type}): {e}")
        return LLMHealthResponse(
            status="error",
            model=model,
            error=str(e)[:500],
            error_type=error_type,
        )


@router.get("/config/model")
async def get_model() -> dict:
    """Get current model and available models. No auth required."""
    return {
        "current": session_manager.config.model_name,
        "available": AVAILABLE_MODELS,
    }


@router.get("/domain-templates")
async def get_domain_templates() -> list[dict[str, Any]]:
    """Return backend-authoritative built-in domain templates."""
    return [template.to_api_dict() for template in list_domain_templates()]


@router.get("/auto-finetune/config")
async def get_auto_finetune_config() -> dict[str, Any]:
    """Return non-secret auto fine-tuning readiness and guardrail config."""
    resolved = resolve_auto_finetune_config(session_manager.config)
    return {
        "provider_id": resolved.provider_id,
        "cost_cap_usd": resolved.cost_cap_usd,
        "max_retries": resolved.max_retries,
        "approval_required": False,
        "credential_readiness": credential_readiness(os.environ),
    }


_TITLE_STRIP_CHARS = str.maketrans("", "", "`*_~#[]()")


def _fallback_title(text: str | None) -> str:
    fallback = text.strip() if isinstance(text, str) else ""
    if not fallback:
        return "New chat"
    return fallback[:40].rstrip() + "…" if len(fallback) > 40 else fallback


def _clean_title_candidate(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    title = value.strip().strip('"').strip("'")
    title = title.translate(_TITLE_STRIP_CHARS).strip()
    if len(title) > 50:
        title = title[:50].rstrip() + "…"
    return title


def _compose_submitted_text(text: Any, context: Any) -> str:
    submitted_text = str(text)
    context_text = render_workflow_context(context)
    if not context_text:
        return submitted_text

    return f"{context_text}\n\nUser request:\n{submitted_text}"


def _stored_workflow_context(
    agent_session: AgentSession,
) -> dict[str, str | None] | None:
    metadata = normalize_workflow_metadata(
        {
            "domain_id": getattr(agent_session, "domain_id", None),
            "provider_id": getattr(agent_session, "provider_id", None),
            "dataset_repo": getattr(agent_session, "dataset_repo", None),
        }
    )
    if (
        metadata["dataset_repo"]
        or metadata["domain_id"] != "generic"
        or metadata["provider_id"] != "hf-jobs"
    ):
        return metadata
    return None


@router.post("/title")
async def generate_title(
    request: SubmitRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Generate a short title for a chat session based on the first user message.

    Always uses gpt-oss-120b via Cerebras on the HF router. The tab headline
    renders as plain text, so the model is told to avoid markdown and any
    stray formatting characters are stripped before returning. gpt-oss is a
    reasoning model — reasoning_effort=low keeps the reasoning budget small
    so the 60-token output budget isn't consumed before the title is written.
    """
    api_key = (
        os.environ.get("INFERENCE_TOKEN")
        or (user.get("hf_token") if isinstance(user, dict) else None)
        or os.environ.get("HF_TOKEN")
    )
    try:
        response = await acompletion(
            # Double openai/ prefix: LiteLLM strips the first as its provider
            # prefix, leaving the HF model id on the wire for the router.
            model="openai/openai/gpt-oss-120b:cerebras",
            api_base="https://router.huggingface.co/v1",
            api_key=api_key,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a very short title (max 6 words) for a chat conversation "
                        "that starts with the following user message. "
                        "Reply with ONLY the title in plain text. "
                        "Do NOT use markdown, backticks, asterisks, quotes, brackets, or any "
                        "formatting characters. No punctuation at the end."
                    ),
                },
                {"role": "user", "content": request.text[:500]},
            ],
            max_tokens=60,
            temperature=0.3,
            timeout=10,
            reasoning_effort="low",
        )
        title = _clean_title_candidate(response.choices[0].message.content)
        return {"title": title or _fallback_title(request.text)}
    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        return {"title": _fallback_title(request.text)}


@router.post("/session", response_model=SessionResponse)
async def create_session(
    request: Request, user: dict = Depends(get_current_user)
) -> SessionResponse:
    """Create a new agent session bound to the authenticated user.

    The user's HF access token is extracted from the Authorization header
    and stored in the session so that tools (e.g. hf_jobs) can act on
    behalf of the user.

    Optional body ``{"model"?: <id>}`` selects the session's LLM; unknown
    ids are rejected (400). The Claude-quota gate runs at message-submit
    time, not here — spinning up an Opus session to look around is free.

    Returns 503 if the server or user has reached the session limit.
    """
    # Extract the user's HF token (Bearer header, HttpOnly cookie, or env var)
    hf_token = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        hf_token = auth_header[7:]
    if not hf_token:
        hf_token = request.cookies.get("hf_access_token")
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")

    # Optional model override. Empty body falls back to the config default.
    model: str | None = None
    metadata = normalize_workflow_metadata({})
    try:
        body = await request.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        model = body.get("model")
        metadata = normalize_workflow_metadata(body)

    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model and model not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    # Opus is gated to HF staff (PR #63). Only fires when the resolved model
    # is Anthropic; free models pass through.
    resolved_model = model or session_manager.config.model_name
    await _require_hf_for_anthropic(request, resolved_model)

    try:
        session_id = await session_manager.create_session(
            user_id=user["user_id"],
            hf_token=hf_token,
            model=model,
            domain_id=metadata["domain_id"],
            provider_id=metadata["provider_id"],
            dataset_repo=metadata["dataset_repo"],
        )
    except SessionCapacityError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return SessionResponse(session_id=session_id, ready=True)


@router.post("/session/restore-summary", response_model=SessionResponse)
async def restore_session_summary(
    request: Request, body: dict, user: dict = Depends(get_current_user)
) -> SessionResponse:
    """Create a new session seeded with a summary of the caller's prior
    conversation. The client sends its cached messages; we run the standard
    summarization prompt on them and drop the result into the new
    session's context as a user-role system note.

    Optional ``"model"`` in the body overrides the session's LLM. The
    Claude-quota gate runs at message-submit time, not here.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="Missing 'messages' array")

    hf_token = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        hf_token = auth_header[7:]
    if not hf_token:
        hf_token = request.cookies.get("hf_access_token")
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")

    model = body.get("model")
    metadata = normalize_workflow_metadata(body)
    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model and model not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    resolved_model = model or session_manager.config.model_name
    await _require_hf_for_anthropic(request, resolved_model)

    try:
        session_id = await session_manager.create_session(
            user_id=user["user_id"],
            hf_token=hf_token,
            model=model,
            domain_id=metadata["domain_id"],
            provider_id=metadata["provider_id"],
            dataset_repo=metadata["dataset_repo"],
        )
    except SessionCapacityError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        summarized = await session_manager.seed_from_summary(session_id, messages)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("seed_from_summary failed")
        raise HTTPException(status_code=500, detail=f"Summary failed: {e}")

    logger.info(
        f"Seeded session {session_id} for {user.get('username', 'unknown')} "
        f"(summary of {summarized} messages)"
    )
    return SessionResponse(session_id=session_id, ready=True)


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str, request: Request, user: dict = Depends(get_current_user)
) -> SessionInfo:
    """Get session information. Only accessible by the session owner."""
    await _check_session_access(session_id, user, request)
    info = session_manager.get_session_info(session_id)
    return SessionInfo(**info)


@router.post("/session/{session_id}/metadata")
async def update_session_metadata(
    session_id: str,
    body: dict,
    request: Request,
    user: dict = Depends(get_current_user),
) -> dict[str, str | None]:
    """Persist backend-normalized workflow metadata for an active session."""
    await _check_session_access(session_id, user, request)
    metadata = normalize_workflow_metadata(body)
    success = await session_manager.update_session_metadata(
        session_id,
        domain_id=metadata["domain_id"],
        provider_id=metadata["provider_id"],
        dataset_repo=metadata["dataset_repo"],
    )
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {
        "domain_id": metadata["domain_id"],
        "provider_id": metadata["provider_id"],
        "dataset_repo": metadata["dataset_repo"],
    }


@router.post("/session/{session_id}/model")
async def set_session_model(
    session_id: str,
    body: dict,
    request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    """Switch the active model for a single session (tab-scoped).

    Takes effect on the next LLM call in that session — other sessions
    (including other browser tabs) are unaffected. Model switches don't
    charge quota — the Claude-quota gate only fires at message-submit time.

    Switching TO an Anthropic model requires HF org membership (PR #63);
    free-model switches are unrestricted.
    """
    await _check_session_access(session_id, user, request)
    model_id = body.get("model")
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing 'model' field")
    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model_id not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
    await _require_hf_for_anthropic(request, model_id)
    agent_session = session_manager.sessions.get(session_id)
    if not agent_session:
        raise HTTPException(status_code=404, detail="Session not found")
    agent_session.session.update_model(model_id)
    await session_manager.persist_session_snapshot(agent_session, runtime_state="idle")
    logger.info(
        f"Session {session_id} model → {model_id} "
        f"(by {user.get('username', 'unknown')})"
    )
    return {"session_id": session_id, "model": model_id}


@router.get("/user/quota")
async def get_user_quota(user: dict = Depends(get_current_user)) -> dict:
    """Return the user's plan tier and today's Claude-session quota state."""
    plan = user.get("plan", "free")
    used = await user_quotas.get_claude_used_today(user["user_id"])
    cap = user_quotas.daily_cap_for(plan)
    return {
        "plan": plan,
        "claude_used_today": used,
        "claude_daily_cap": cap,
        "claude_remaining": max(0, cap - used),
    }


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(user: dict = Depends(get_current_user)) -> list[SessionInfo]:
    """List sessions belonging to the authenticated user."""
    sessions = await session_manager.list_sessions(user_id=user["user_id"])
    return [SessionInfo(**s) for s in sessions]


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str, request: Request, user: dict = Depends(get_current_user)
) -> dict:
    """Delete a session. Only accessible by the session owner."""
    await _check_session_delete_access(session_id, user)
    success = await session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.post("/submit")
async def submit_input(
    request: SubmitRequest,
    http_request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    """Submit user input to a session. Only accessible by the session owner."""
    agent_session = await _check_session_access(request.session_id, user, http_request)
    await _enforce_claude_quota(user, agent_session)
    success = await session_manager.submit_user_input(request.session_id, request.text)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "submitted", "session_id": request.session_id}


@router.post("/approve")
async def submit_approval(
    request: ApprovalRequest,
    http_request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    """Submit tool approvals to a session. Only accessible by the session owner."""
    await _check_session_access(request.session_id, user, http_request)
    approvals = [
        {
            "tool_call_id": a.tool_call_id,
            "approved": a.approved,
            "feedback": a.feedback,
            "edited_script": a.edited_script,
        }
        for a in request.approvals
    ]
    success = await session_manager.submit_approval(request.session_id, approvals)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "submitted", "session_id": request.session_id}


@router.post("/chat/{session_id}")
async def chat_sse(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """SSE endpoint: submit input or approval, then stream events until turn ends."""
    agent_session = await _check_session_access(session_id, user, request)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")

    # Parse body
    body = await request.json()

    # Subscribe BEFORE submitting so we never miss events — even if the
    # agent loop processes the submission before this coroutine continues.
    broadcaster = agent_session.broadcaster
    sub_id, event_queue = broadcaster.subscribe()

    # Submit the operation
    text = body.get("text")
    context = body.get("context")
    approvals = body.get("approvals")
    effective_context = context
    if text is not None and isinstance(context, (dict, str)):
        metadata = normalize_workflow_metadata(context)
        await session_manager.update_session_metadata(
            session_id,
            domain_id=metadata["domain_id"],
            provider_id=metadata["provider_id"],
            dataset_repo=metadata["dataset_repo"],
        )
        effective_context = metadata
    elif text is not None:
        effective_context = _stored_workflow_context(agent_session)

    # Gate user-message sends against the daily Claude quota. Approvals are
    # continuations of an in-progress turn — the session was already charged
    # on its first message, so we skip the gate there.
    if text is not None and not approvals:
        try:
            await _enforce_claude_quota(user, agent_session)
        except HTTPException:
            broadcaster.unsubscribe(sub_id)
            raise

    try:
        if approvals:
            formatted = [
                {
                    "tool_call_id": a["tool_call_id"],
                    "approved": a["approved"],
                    "feedback": a.get("feedback"),
                    "edited_script": a.get("edited_script"),
                }
                for a in approvals
            ]
            success = await session_manager.submit_approval(session_id, formatted)
        elif text is not None:
            submitted_text = _compose_submitted_text(text, effective_context)
            success = await session_manager.submit_user_input(
                session_id, submitted_text
            )
        else:
            broadcaster.unsubscribe(sub_id)
            raise HTTPException(
                status_code=400, detail="Must provide 'text' or 'approvals'"
            )

        if not success:
            broadcaster.unsubscribe(sub_id)
            raise HTTPException(status_code=404, detail="Session not found or inactive")
    except HTTPException:
        raise
    except Exception:
        broadcaster.unsubscribe(sub_id)
        raise

    return _sse_response(broadcaster, event_queue, sub_id)


# ---------------------------------------------------------------------------
# Shared SSE helpers
# ---------------------------------------------------------------------------
_TERMINAL_EVENTS = {
    "turn_complete",
    "approval_required",
    "error",
    "interrupted",
    "shutdown",
}
_SSE_KEEPALIVE_SECONDS = 15


def _last_event_seq(request: Request) -> int | None:
    raw = request.headers.get("last-event-id") or request.query_params.get("after")
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _format_sse(msg: dict[str, Any]) -> str:
    seq = msg.get("seq")
    body = {"event_type": msg.get("event_type"), "data": msg.get("data") or {}}
    if seq is not None:
        body["seq"] = seq
        return f"id: {seq}\ndata: {json.dumps(body)}\n\n"
    return f"data: {json.dumps(body)}\n\n"


def _sse_response(
    broadcaster,
    event_queue,
    sub_id,
    *,
    replay_events: list[dict[str, Any]] | None = None,
    after_seq: int = 0,
) -> StreamingResponse:
    """Build a StreamingResponse that drains *event_queue* as SSE,
    sending keepalive comments every 15 s to prevent proxy timeouts."""

    async def event_generator():
        last_sent_seq = after_seq
        try:
            for msg in replay_events or []:
                seq = msg.get("seq")
                if isinstance(seq, int) and seq <= last_sent_seq:
                    continue
                yield _format_sse(msg)
                if isinstance(seq, int):
                    last_sent_seq = seq
                if msg.get("event_type", "") in _TERMINAL_EVENTS:
                    return

            while True:
                try:
                    msg = await asyncio.wait_for(
                        event_queue.get(), timeout=_SSE_KEEPALIVE_SECONDS
                    )
                except asyncio.TimeoutError:
                    # SSE comment — ignored by parsers, keeps connection alive
                    yield ": keepalive\n\n"
                    continue
                event_type = msg.get("event_type", "")
                seq = msg.get("seq")
                if isinstance(seq, int) and seq <= last_sent_seq:
                    continue
                yield _format_sse(msg)
                if isinstance(seq, int):
                    last_sent_seq = seq
                if event_type in _TERMINAL_EVENTS:
                    break
        finally:
            broadcaster.unsubscribe(sub_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/events/{session_id}")
async def subscribe_events(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """Subscribe to events for a running session without submitting new input.

    Used by the frontend to re-attach after a connection drop (e.g. screen
    sleep).  Returns 404 if the session isn't active or isn't processing.
    """
    agent_session = await _check_session_access(session_id, user, request)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")

    broadcaster = agent_session.broadcaster
    sub_id, event_queue = broadcaster.subscribe()
    after_seq = _last_event_seq(request)
    replay_events = None
    if after_seq is not None:
        replay_events = await session_manager._store().load_events_after(
            session_id, after_seq
        )
    return _sse_response(
        broadcaster,
        event_queue,
        sub_id,
        replay_events=replay_events,
        after_seq=after_seq or 0,
    )


@router.post("/interrupt/{session_id}")
async def interrupt_session(
    session_id: str, request: Request, user: dict = Depends(get_current_user)
) -> dict:
    """Interrupt the current operation in a session."""
    await _check_session_access(session_id, user, request)
    success = await session_manager.interrupt(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "interrupted", "session_id": session_id}


@router.get("/session/{session_id}/messages")
async def get_session_messages(
    session_id: str, request: Request, user: dict = Depends(get_current_user)
) -> list[dict]:
    """Return the session's message history from memory."""
    agent_session = await _check_session_access(session_id, user, request)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return [msg.model_dump() for msg in agent_session.session.context_manager.items]


@router.post("/undo/{session_id}")
async def undo_session(
    session_id: str, request: Request, user: dict = Depends(get_current_user)
) -> dict:
    """Undo the last turn in a session."""
    await _check_session_access(session_id, user, request)
    success = await session_manager.undo(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "undo_requested", "session_id": session_id}


@router.post("/truncate/{session_id}")
async def truncate_session(
    session_id: str,
    body: TruncateRequest,
    request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    """Truncate conversation to before a specific user message."""
    await _check_session_access(session_id, user, request)
    success = await session_manager.truncate(session_id, body.user_message_index)
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Session not found, inactive, or message index out of range",
        )
    return {"status": "truncated", "session_id": session_id}


@router.post("/compact/{session_id}")
async def compact_session(
    session_id: str, request: Request, user: dict = Depends(get_current_user)
) -> dict:
    """Compact the context in a session."""
    await _check_session_access(session_id, user, request)
    success = await session_manager.compact(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "compact_requested", "session_id": session_id}


@router.post("/shutdown/{session_id}")
async def shutdown_session(
    session_id: str, request: Request, user: dict = Depends(get_current_user)
) -> dict:
    """Shutdown a session."""
    await _check_session_access(session_id, user, request)
    success = await session_manager.shutdown_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "shutdown_requested", "session_id": session_id}
