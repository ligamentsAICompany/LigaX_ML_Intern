"""Dataset upload validation and Hugging Face Hub operations."""

import os
import json
import re
import tempfile
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from fastapi import HTTPException, Request, UploadFile
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from agent.core.dataset_inspection import (
    build_dataset_profile,
    extract_local_dataset_rows,
    inspect_local_dataset,
)
from agent.core import telemetry
from agent.core.redact import scrub_string

ALLOWED_DATASET_EXTENSIONS = {
    ".csv",
    ".docx",
    ".json",
    ".jsonl",
    ".parquet",
    ".pdf",
    ".xlsx",
}
PRIMARY_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".csv", ".xlsx"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_PROFILE_UPLOAD_BYTES = 10 * 1024 * 1024
_SESSION_REPO_RE = re.compile(r"(?:^|-)session-([A-Za-z0-9]{4,16})$")
_SLUG_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FILENAME_STOP_WORDS = {
    "data",
    "dataset",
    "file",
    "files",
    "note",
    "notes",
    "pair",
    "pairs",
    "reference",
    "references",
    "upload",
    "uploads",
}


def safe_exception_detail(prefix: str, exc: Exception) -> str:
    """Keep upload errors actionable without exposing credentials."""
    return f"{prefix}: {scrub_string(str(exc))[:300]}"


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


def get_hf_username(token: str) -> str | None:
    """Return the authenticated Hub username for an already-validated token."""
    whoami = HfApi(token=token).whoami()
    name = whoami.get("name") if isinstance(whoami, dict) else None
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def resolve_upload_repo_id(
    repo_id: str, *, hf_username: str | None, app_username: str | None = None
) -> str:
    """Resolve local placeholder namespaces to the authenticated Hub owner."""
    normalized = repo_id.strip().removeprefix("datasets/").strip("/")
    if not normalized or not hf_username:
        return normalized

    if "/" not in normalized:
        return f"{hf_username}/{normalized}"

    namespace, repo_name = normalized.split("/", 1)
    placeholder_namespaces = {"dev", "ml-intern"}
    if app_username:
        placeholder_namespaces.add(app_username)
    if namespace in placeholder_namespaces and namespace != hf_username:
        return f"{hf_username}/{repo_name}"
    return normalized


def friendly_upload_repo_id_from_filenames(
    repo_id: str, filenames: list[str | None]
) -> str:
    """Replace placeholder session repo names with a safe filename-derived slug."""
    normalized = repo_id.strip().removeprefix("datasets/").strip("/")
    if "/" not in normalized:
        return normalized

    namespace, repo_name = normalized.split("/", 1)
    suffix_match = _SESSION_REPO_RE.search(repo_name)
    if not suffix_match:
        return normalized

    slug = _upload_filename_slug(filenames)
    if not slug:
        return normalized

    suffix = suffix_match.group(1).lower()
    return f"{namespace}/{slug[:80].strip('-')}-{suffix}"


def _upload_filename_slug(filenames: list[str | None]) -> str:
    parts: list[str] = []
    cleaned = [_filename_tokens(filename) for filename in filenames if filename]
    cleaned = [tokens for tokens in cleaned if tokens]
    if not cleaned:
        return ""

    if len(cleaned) == 1:
        stem_tokens, extension = cleaned[0]
        parts.extend(stem_tokens)
        if extension:
            parts.append(extension)
    else:
        for stem_tokens, _extension in cleaned[:4]:
            token = next(
                (item for item in stem_tokens if item not in _FILENAME_STOP_WORDS),
                stem_tokens[0],
            )
            parts.append(token)
        parts.append("bundle")

    return "-".join(_dedupe_ordered(parts)).strip("-")


def _filename_tokens(filename: str | None) -> tuple[list[str], str] | None:
    raw = str(PurePosixPath(PureWindowsPath((filename or "").strip()).name).name)
    stem, extension = os.path.splitext(raw)
    tokens = _SLUG_TOKEN_RE.findall(stem.lower())
    extension_token = extension.lower().lstrip(".")
    if not tokens and not extension_token:
        return None
    return (tokens or ["dataset"], extension_token)


def _dedupe_ordered(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        if part and part not in seen:
            unique.append(part)
            seen.add(part)
    return unique


def sanitize_dataset_filename(filename: str | None) -> str:
    """Return a Hub-safe filename basename while preserving supported extensions."""
    raw = (filename or "").strip()
    raw = str(PurePosixPath(PureWindowsPath(raw).name).name)
    if not raw:
        raw = "dataset.csv"

    stem, extension = os.path.splitext(raw)
    extension = extension.lower()
    if extension not in ALLOWED_DATASET_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only .pdf, .docx, .csv, .xlsx uploads are supported for "
                "conversion; .json, .jsonl, and .parquet remain supported "
                "dataset uploads."
            ),
        )

    safe_stem = "".join(
        char if char.isalnum() or char in "._-" else "-" for char in stem
    )
    safe_stem = safe_stem.strip(".-_") or "dataset"
    max_stem_len = 96 - len(extension)
    safe_stem = safe_stem[:max_stem_len].strip(".-_") or "dataset"
    return f"{safe_stem}{extension}"


async def validate_dataset_file(file: UploadFile) -> tuple[str, bytes]:
    """Validate supported type, non-empty body, and upload size."""
    safe_filename = sanitize_dataset_filename(file.filename)
    size = get_upload_size(file)
    if size is not None:
        if size <= 0:
            raise HTTPException(
                status_code=400, detail="Uploaded dataset file is empty."
            )
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413, detail="Dataset upload exceeds the 100 MB limit."
            )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    size = len(contents)
    if size <= 0:
        raise HTTPException(status_code=400, detail="Uploaded dataset file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail="Dataset upload exceeds the 100 MB limit."
        )
    return safe_filename, contents


async def validate_dataset_files(files: list[UploadFile]) -> list[dict[str, Any]]:
    """Validate a multi-file upload and return unique safe filenames with bytes."""

    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one dataset file.")

    validated: list[dict[str, Any]] = []
    used_filenames: set[str] = set()
    total_size = 0
    for file in files:
        safe_filename, contents = await validate_dataset_file(file)
        safe_filename = unique_safe_filename(safe_filename, used_filenames)
        used_filenames.add(safe_filename)
        total_size += len(contents)
        if total_size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413, detail="Dataset upload exceeds the 100 MB limit."
            )
        validated.append(
            {
                "filename": safe_filename,
                "contents": contents,
                "size_bytes": len(contents),
                "format": os.path.splitext(safe_filename)[1].lstrip(".").lower(),
            }
        )
    return validated


def unique_safe_filename(filename: str, used_filenames: set[str]) -> str:
    """Avoid collisions when multiple uploaded files sanitize to the same basename."""

    if filename not in used_filenames:
        return filename
    stem, extension = os.path.splitext(filename)
    for index in range(2, 10_000):
        candidate = f"{stem}-{index}{extension}"
        if candidate not in used_filenames:
            return candidate
    raise HTTPException(status_code=400, detail="Too many duplicate filenames.")


def profile_dataset_contents(*, safe_filename: str, contents: bytes) -> dict[str, Any]:
    """Profile validated upload bytes without exposing the temporary filesystem path."""

    if len(contents) > MAX_PROFILE_UPLOAD_BYTES:
        raise ValueError(
            "Upload is larger than the profiling limit; skipping best-effort "
            "dataset profiling to avoid large memory expansion."
        )

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=f"_{safe_filename}", delete=False
        ) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        profile = inspect_local_dataset(tmp_path)
        profile["source"]["path"] = safe_filename
        return profile
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_upload_size(file: UploadFile) -> int | None:
    """Return upload size without reading the body when the backing file supports it."""
    explicit_size = getattr(file, "size", None)
    if isinstance(explicit_size, int):
        return explicit_size

    backing_file = getattr(file, "file", None)
    if backing_file is None:
        return None

    try:
        original_position = backing_file.tell()
        backing_file.seek(0, os.SEEK_END)
        size = backing_file.tell()
        backing_file.seek(original_position, os.SEEK_SET)
    except (AttributeError, OSError):
        return None

    return size


async def upload_dataset_file(
    *, file: UploadFile, repo_id: str, token: str
) -> dict[str, Any]:
    """Validate and upload a dataset file to a private HF dataset repo."""
    safe_filename, contents = await validate_dataset_file(file)
    return upload_dataset_contents(
        safe_filename=safe_filename,
        contents=contents,
        repo_id=repo_id,
        token=token,
    )


async def upload_dataset_files(
    *, files: list[UploadFile], repo_id: str, token: str
) -> dict[str, Any]:
    """Validate, convert, and upload multiple files plus a JSONL dataset artifact."""

    validated_files = await validate_dataset_files(files)
    rows, dataset_profile = convert_uploaded_files(validated_files)
    train_contents = jsonl_bytes(rows)

    api = create_private_dataset_repo(repo_id=repo_id, token=token)
    for item in validated_files:
        upload_bytes_to_repo(
            api=api,
            repo_id=repo_id,
            path_in_repo=f"raw/{item['filename']}",
            contents=item["contents"],
            safe_filename=item["filename"],
        )
    upload_bytes_to_repo(
        api=api,
        repo_id=repo_id,
        path_in_repo="train.jsonl",
        contents=train_contents,
        safe_filename="train.jsonl",
    )

    telemetry.record_dataset_upload_sync(
        repo_id=repo_id,
        filename="train.jsonl",
        size_bytes=sum(item["size_bytes"] for item in validated_files),
        status="success",
    )
    return {
        "dataset_id": repo_id,
        "filename": "train.jsonl",
        "files": [
            {
                "filename": item["filename"],
                "format": item["format"],
                "size_bytes": item["size_bytes"],
            }
            for item in validated_files
        ],
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "dataset_profile": dataset_profile,
    }


def convert_uploaded_files(
    validated_files: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract uploaded files into JSONL rows and profile the combined dataset."""

    combined_rows: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []
    tmp_paths: list[str] = []
    try:
        for item in validated_files:
            with tempfile.NamedTemporaryFile(
                suffix=f"_{item['filename']}", delete=False
            ) as tmp:
                tmp.write(item["contents"])
                tmp_paths.append(tmp.name)
                tmp_path = tmp.name
            extracted_rows, _source = extract_local_dataset_rows(
                tmp_path,
                source_filename=item["filename"],
            )
            combined_rows.extend(dict(row) for row in extracted_rows)
            file_summaries.append(
                {
                    "filename": item["filename"],
                    "format": item["format"],
                    "size_bytes": item["size_bytes"],
                    "row_count": len(extracted_rows),
                }
            )
    finally:
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    source = {
        "type": "uploaded_files",
        "format": "jsonl",
        "files": file_summaries,
        "formats": sorted({item["format"] for item in validated_files}),
    }
    profile = build_dataset_profile(rows=combined_rows, source=source)
    return combined_rows, profile


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Serialize extracted rows for Dataset Viewer and downstream pipelines."""

    return (
        "\n".join(json.dumps(row, ensure_ascii=True, default=str) for row in rows)
        + "\n"
    ).encode("utf-8")


def create_private_dataset_repo(*, repo_id: str, token: str) -> HfApi:
    """Create or reuse a private dataset repo after validating Hub auth."""

    api = HfApi(token=token)
    try:
        api.whoami()
        api.create_repo(
            repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True
        )
    except HfHubHTTPError as exc:
        raise_hub_auth_error(exc, "create dataset repo", repo_id)
        raise HTTPException(
            status_code=400, detail=safe_exception_detail("Failed to create repo", exc)
        ) from exc
    except Exception as exc:
        raise_hub_auth_error_from_message(exc, "create dataset repo", repo_id)
        raise HTTPException(
            status_code=400, detail=safe_exception_detail("Failed to create repo", exc)
        ) from exc
    return api


def upload_bytes_to_repo(
    *,
    api: HfApi,
    repo_id: str,
    path_in_repo: str,
    contents: bytes,
    safe_filename: str,
) -> None:
    """Upload bytes through a temp file so failed uploads cannot leak local paths."""

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=f"_{safe_filename}", delete=False
        ) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
    except HfHubHTTPError as exc:
        raise_hub_auth_error(exc, "upload dataset file", repo_id)
        raise HTTPException(
            status_code=500, detail=safe_exception_detail("Upload failed", exc)
        ) from exc
    except Exception as exc:
        raise_hub_auth_error_from_message(exc, "upload dataset file", repo_id)
        raise HTTPException(
            status_code=500, detail=safe_exception_detail("Upload failed", exc)
        ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def upload_dataset_contents(
    *, safe_filename: str, contents: bytes, repo_id: str, token: str
) -> dict[str, Any]:
    """Upload already-validated dataset bytes to a private HF dataset repo."""
    dataset_profile: dict[str, Any] | None = None
    dataset_profile_error: str | None = None
    try:
        dataset_profile = profile_dataset_contents(
            safe_filename=safe_filename,
            contents=contents,
        )
    except Exception as exc:
        dataset_profile_error = safe_exception_detail("Dataset profiling skipped", exc)

    api = HfApi(token=token)

    try:
        api.whoami()
        api.create_repo(
            repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True
        )
    except HfHubHTTPError as exc:
        raise_hub_auth_error(exc, "create dataset repo", repo_id)
        raise HTTPException(
            status_code=400, detail=safe_exception_detail("Failed to create repo", exc)
        ) from exc
    except Exception as exc:
        raise_hub_auth_error_from_message(exc, "create dataset repo", repo_id)
        raise HTTPException(
            status_code=400, detail=safe_exception_detail("Failed to create repo", exc)
        ) from exc

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=f"_{safe_filename}", delete=False
        ) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=safe_filename,
            repo_id=repo_id,
            repo_type="dataset",
        )
    except HfHubHTTPError as exc:
        raise_hub_auth_error(exc, "upload dataset file", repo_id)
        raise HTTPException(
            status_code=500, detail=safe_exception_detail("Upload failed", exc)
        ) from exc
    except Exception as exc:
        raise_hub_auth_error_from_message(exc, "upload dataset file", repo_id)
        raise HTTPException(
            status_code=500, detail=safe_exception_detail("Upload failed", exc)
        ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    telemetry.record_dataset_upload_sync(
        repo_id=repo_id,
        filename=safe_filename,
        size_bytes=len(contents),
        status="success",
    )
    payload: dict[str, Any] = {
        "dataset_id": repo_id,
        "filename": safe_filename,
        "url": f"https://huggingface.co/datasets/{repo_id}",
    }
    if dataset_profile is not None:
        payload["dataset_profile"] = dataset_profile
    if dataset_profile_error:
        payload["dataset_profile_error"] = dataset_profile_error
    return payload


def raise_hub_auth_error(
    exc: Exception, operation: str, repo_id: str | None = None
) -> None:
    """Map HF Hub auth/permission failures to actionable API errors."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    detail_suffix = f" for repo '{repo_id}'" if repo_id else ""
    if status == 401:
        raise HTTPException(
            status_code=401,
            detail=(
                f"Hub authentication failed while attempting to {operation}{detail_suffix}. "
                "Use a valid Hugging Face write token (HF_TOKEN) or log in via /auth/login."
            ),
        ) from exc
    if status == 403:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Hub permission denied while attempting to {operation}{detail_suffix}. "
                "Ensure this account can create/write repos in the target namespace."
            ),
        ) from exc


def raise_hub_auth_error_from_message(
    exc: Exception, operation: str, repo_id: str | None = None
) -> None:
    """Best-effort auth/permission mapping for non-HfHub exceptions."""
    text = str(exc).lower()
    detail_suffix = f" for repo {repo_id!r}" if repo_id else ""
    if "401 unauthorized" in text or "invalid username or password" in text:
        raise HTTPException(
            status_code=401,
            detail=(
                f"Hub authentication failed while attempting to {operation}{detail_suffix}. "
                "Use a valid Hugging Face write token (HF_TOKEN) or log in via /auth/login."
            ),
        ) from exc
    if "403 forbidden" in text or "permission" in text:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Hub permission denied while attempting to {operation}{detail_suffix}. "
                "Ensure this account can create/write repos in the target namespace."
            ),
        ) from exc
