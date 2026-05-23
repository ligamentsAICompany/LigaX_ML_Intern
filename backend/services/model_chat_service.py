"""Model-chat token resolution, auto-fix, and SSE streaming behavior."""

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from huggingface_hub import HfApi, ModelCard

from agent.core import telemetry
from agent.core.redact import scrub_string

logger = logging.getLogger(__name__)


def resolve_hf_token(request: Request) -> str:
    """Resolve and validate the HF token using the platform route's existing order."""
    candidates: list[str] = []
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token:
            candidates.append(token)

    cookie_token = request.cookies.get("hf_access_token", "").strip()
    if cookie_token:
        candidates.append(cookie_token)

    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        candidates.append(env_token)

    checked: set[str] = set()
    for token in candidates:
        if token in checked:
            continue
        checked.add(token)
        try:
            HfApi(token=token).whoami()
            return token
        except Exception:
            continue

    raise HTTPException(
        status_code=401,
        detail=(
            "No valid HF auth token found (Bearer/cookie/HF_TOKEN). Please log in via "
            "/auth/login or configure a valid HF_TOKEN."
        ),
    )


def parse_model_chat_body(body: dict[str, Any]) -> tuple[str, list[Any]]:
    """Validate the route body without changing existing error messages."""
    model_id = str(body.get("model_id", "")).strip()
    messages = body.get("messages", [])

    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    if not messages:
        raise HTTPException(status_code=400, detail="messages array is required")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages array is required")
    return model_id, messages


async def auto_fix_chat_model(token: str, model_id: str) -> None:
    """Best-effort background patch for missing chat template/pipeline metadata."""
    try:
        api = HfApi(token=token)
        info = api.model_info(model_id)

        base_model = None
        for tag in getattr(info, "tags", []):
            if tag.startswith("base_model:"):
                potential_base = tag.split(":", 1)[1]
                if potential_base != model_id and "finetune" not in potential_base:
                    base_model = potential_base
                    break

        if not base_model:
            for tag in getattr(info, "tags", []):
                if tag.startswith("base_model:finetune:"):
                    base_model = tag.split("base_model:finetune:")[1]
                    break

        if not base_model:
            logger.info("Auto-fix failed for %s: No base_model tag found.", model_id)
            return

        logger.info(
            "Auto-fix triggered for %s. Base model is %s.", model_id, base_model
        )
        tokenizer_files = [
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "special_tokens_map.json",
        ]

        base_files = api.list_repo_files(base_model)
        files_to_copy = [file for file in tokenizer_files if file in base_files]

        for filename in files_to_copy:
            try:
                path = api.hf_hub_download(
                    repo_id=base_model, filename=filename, revision="main"
                )
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=filename,
                    repo_id=model_id,
                    commit_message=f"Auto-fix: copied {filename} from {base_model}",
                )
            except Exception as exc:
                logger.warning(
                    "Failed to copy %s from %s: %s", filename, base_model, exc
                )

        try:
            card = ModelCard.load(model_id, token=token)
            card.data.pipeline_tag = "text-generation"
            card.push_to_hub(model_id, token=token)
        except Exception as exc:
            logger.warning("Failed to push updated ModelCard for %s: %s", model_id, exc)

        logger.info("Auto-fix complete for %s.", model_id)
    except Exception as exc:
        logger.error("Error in auto_fix_chat_model for %s: %s", model_id, exc)


async def stream_model_chat(
    *,
    token: str,
    model_id: str,
    messages: list[Any],
    async_client_factory: Callable[..., Any] = httpx.AsyncClient,
) -> AsyncIterator[str]:
    """Yield HF Inference API SSE chunks in the route-compatible shape."""
    telemetry.record_model_chat_sync(
        model_id=model_id,
        message_count=len(messages),
        status="started",
    )
    inference_url = (
        f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions"
    )
    async with async_client_factory(timeout=120) as client:
        try:
            async with client.stream(
                "POST",
                inference_url,
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "model": model_id,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": 512,
                    "temperature": 0.7,
                },
            ) as resp:
                if resp.status_code != 200:
                    body_text = await resp.aread()
                    err_text = body_text.decode(errors="replace")
                    safe_err_text = scrub_string(err_text)[:300]
                    err_msg: dict[str, Any] = {
                        "error": f"HF API {resp.status_code}",
                        "detail": safe_err_text,
                    }

                    is_chat_error = (
                        "is not a chat model" in err_text
                        or "model_not_supported" in err_text
                        or "not supported by any provider" in err_text
                    )
                    if resp.status_code in (400, 422) and is_chat_error:
                        asyncio.create_task(auto_fix_chat_model(token, model_id))
                        err_msg = {
                            "error": "HF API \u2014 Model is missing chat template",
                            "detail": safe_err_text,
                            "auto_fix": True,
                            "auto_fix_message": (
                                "\u26a1 Auto-Fix In Progress \u2014 The server is patching the "
                                "chat template from the base model. Retrying automatically in "
                                "20 seconds\u2026"
                            ),
                            "retry_after": 20,
                        }

                    yield f"data: {json.dumps(err_msg)}\n\n"
                    telemetry.record_model_chat_sync(
                        model_id=model_id,
                        message_count=len(messages),
                        status="error",
                    )
                    return
                async for chunk in resp.aiter_text():
                    yield chunk
                telemetry.record_model_chat_sync(
                    model_id=model_id,
                    message_count=len(messages),
                    status="success",
                )
        except Exception as exc:
            telemetry.record_model_chat_sync(
                model_id=model_id,
                message_count=len(messages),
                status="error",
            )
            yield f"data: {json.dumps({'error': scrub_string(str(exc))[:300]})}\n\n"


def model_chat_response(
    token: str, model_id: str, messages: list[Any]
) -> StreamingResponse:
    """Build the public SSE response for model chat."""
    return StreamingResponse(
        stream_model_chat(token=token, model_id=model_id, messages=messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
