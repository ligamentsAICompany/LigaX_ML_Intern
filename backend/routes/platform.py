"""Platform routes: dataset upload and model inference proxy."""

import logging

from dependencies import get_current_user
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from services import dataset_service, model_chat_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/platform", tags=["platform"])


@router.post("/upload-dataset")
async def upload_dataset(
    request: Request,
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    repo_id: str = Form(...),
    user: dict = Depends(get_current_user),
):
    """Upload one or more dataset files to a private HF dataset repo."""
    token = dataset_service.resolve_hf_token(request)
    hf_username = dataset_service.get_hf_username(token)
    resolved_repo_id = dataset_service.resolve_upload_repo_id(
        repo_id,
        hf_username=hf_username,
        app_username=str(user.get("username") or user.get("user_id") or ""),
    )
    upload_files = list(files or [])
    if upload_files:
        payload = await dataset_service.upload_dataset_files(
            files=upload_files,
            repo_id=resolved_repo_id,
            token=token,
        )
        logger.info(
            "Uploaded %s files to dataset %s", len(upload_files), resolved_repo_id
        )
        return payload
    if file is None:
        raise HTTPException(status_code=400, detail="Upload at least one dataset file.")

    safe_filename, contents = await dataset_service.validate_dataset_file(file)
    payload = dataset_service.upload_dataset_contents(
        safe_filename=safe_filename,
        contents=contents,
        repo_id=resolved_repo_id,
        token=token,
    )
    logger.info("Uploaded %s to dataset %s", payload["filename"], resolved_repo_id)
    return payload


@router.post("/model-chat")
async def model_chat(
    request: Request, body: dict, user: dict = Depends(get_current_user)
):
    """Stream chat completions from a HF Hub model via HF Router."""
    token = model_chat_service.resolve_hf_token(request)
    model_id, messages = model_chat_service.parse_model_chat_body(body)
    return model_chat_service.model_chat_response(token, model_id, messages)
