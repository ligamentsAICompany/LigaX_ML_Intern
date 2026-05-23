# FastAPI Integration Docs (No Chat APIs)

Base URL (current deployment):

`http://35.184.58.132:7860`

OpenAPI UI:

`http://35.184.58.132:7860/docs`

## Auth

This backend accepts auth in this order:

1. `Authorization: Bearer <token>`
2. `hf_access_token` cookie (from `/auth/login`)
3. Server-side `HF_TOKEN` env fallback (for supported routes)

For browser integrations, send cookies:

```js
fetch(url, { credentials: "include" })
```

---

## 1) API Root

### GET `/api`

#### curl
```bash
curl -s "http://35.184.58.132:7860/api"
```

#### Response (200)
```json
{
  "name": "HF Agent API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

---

## 2) Health

### GET `/api/health`

#### curl
```bash
curl -s "http://35.184.58.132:7860/api/health"
```

#### Response (200)
```json
{
  "status": "ok",
  "active_sessions": 1,
  "max_sessions": 50
}
```

### GET `/api/health/llm`

#### curl
```bash
curl -s "http://35.184.58.132:7860/api/health/llm"
```

#### Response (200)
```json
{
  "status": "ok",
  "model": "moonshotai/Kimi-K2.6",
  "error": null,
  "error_type": null
}
```

---

## 3) Auth Endpoints

### GET `/auth/status`

#### curl
```bash
curl -s "http://35.184.58.132:7860/auth/status"
```

#### Response (200)
```json
{
  "auth_enabled": false
}
```

### GET `/auth/login`

Redirects to Hugging Face OAuth when enabled.

#### curl
```bash
curl -i "http://35.184.58.132:7860/auth/login"
```

#### Response
- `302 Found` with `Location: https://huggingface.co/oauth/authorize...`

### GET `/auth/logout`

Clears auth cookie.

#### curl
```bash
curl -i "http://35.184.58.132:7860/auth/logout"
```

#### Response
- `307/302` redirect to `/`

### GET `/auth/me`

Returns current user from auth dependency.

#### curl
```bash
curl -s \
  -H "Authorization: Bearer <HF_TOKEN>" \
  "http://35.184.58.132:7860/auth/me"
```

#### Response (200)
```json
{
  "user_id": "dev",
  "username": "dev",
  "authenticated": true,
  "plan": "org"
}
```

### GET `/auth/org-membership`

#### curl
```bash
curl -s \
  -H "Authorization: Bearer <HF_TOKEN>" \
  "http://35.184.58.132:7860/auth/org-membership"
```

#### Response (200)
```json
{
  "is_member": true
}
```

---

## 4) Platform (Dataset Upload)

### POST `/api/platform/upload-dataset`

Uploads one dataset file and creates repo if needed.

#### curl
```bash
curl -s \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -F "file=@bitext-telco-llm-chatbot-training-dataset.csv" \
  -F "repo_id=ligaments-dev/Qwen-telecom-chatbot-data" \
  "http://35.184.58.132:7860/api/platform/upload-dataset"
```

#### Request (multipart/form-data)
- `file`: file upload (`.csv`, `.json`, `.jsonl`, `.parquet`, etc.)
- `repo_id`: target HF dataset repo id (`org/name`)

#### Response (200)
```json
{
  "dataset_id": "ligaments-dev/Qwen-telecom-chatbot-data",
  "filename": "bitext-telco-llm-chatbot-training-dataset.csv",
  "url": "https://huggingface.co/datasets/ligaments-dev/Qwen-telecom-chatbot-data"
}
```

#### Common Error Responses
```json
{
  "detail": "Hub authentication failed while attempting to create dataset repo for repo 'ligaments-dev/Qwen-telecom-chatbot-data'. Use a valid Hugging Face write token (HF_TOKEN) or log in via /auth/login."
}
```

```json
{
  "detail": "Hub permission denied while attempting to create dataset repo for repo 'ligaments-dev/Qwen-telecom-chatbot-data'. Ensure this account can create/write repos in the target namespace."
}
```

---

## 5) Session + Control APIs (Non-Chat)

### POST `/api/session`

Creates a session.

#### curl
```bash
curl -s \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -d '{"model":"moonshotai/Kimi-K2.6"}' \
  "http://35.184.58.132:7860/api/session"
```

#### Response (200)
```json
{
  "session_id": "3b67f1cb-3ef4-4af4-b12c-1e0f3f2d9abc",
  "ready": true
}
```

### GET `/api/session/{session_id}`

#### curl
```bash
curl -s \
  -H "Authorization: Bearer <HF_TOKEN>" \
  "http://35.184.58.132:7860/api/session/<session_id>"
```

#### Response (200)
```json
{
  "session_id": "3b67f1cb-3ef4-4af4-b12c-1e0f3f2d9abc",
  "created_at": "2026-04-25T17:00:00.000000",
  "is_active": true,
  "is_processing": false,
  "message_count": 0,
  "user_id": "dev",
  "pending_approval": null,
  "model": "moonshotai/Kimi-K2.6"
}
```

### GET `/api/sessions`

#### curl
```bash
curl -s \
  -H "Authorization: Bearer <HF_TOKEN>" \
  "http://35.184.58.132:7860/api/sessions"
```

#### Response (200)
```json
[
  {
    "session_id": "3b67f1cb-3ef4-4af4-b12c-1e0f3f2d9abc",
    "created_at": "2026-04-25T17:00:00.000000",
    "is_active": true,
    "is_processing": false,
    "message_count": 0,
    "user_id": "dev",
    "pending_approval": null,
    "model": "moonshotai/Kimi-K2.6"
  }
]
```

### DELETE `/api/session/{session_id}`

#### curl
```bash
curl -s -X DELETE \
  -H "Authorization: Bearer <HF_TOKEN>" \
  "http://35.184.58.132:7860/api/session/<session_id>"
```

#### Response (200)
```json
{
  "status": "deleted",
  "session_id": "3b67f1cb-3ef4-4af4-b12c-1e0f3f2d9abc"
}
```

### POST `/api/session/{session_id}/model`

#### curl
```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -d '{"model":"moonshotai/Kimi-K2.6"}' \
  "http://35.184.58.132:7860/api/session/<session_id>/model"
```

#### Response (200)
```json
{
  "session_id": "3b67f1cb-3ef4-4af4-b12c-1e0f3f2d9abc",
  "model": "moonshotai/Kimi-K2.6"
}
```

### GET `/api/user/quota`

#### curl
```bash
curl -s \
  -H "Authorization: Bearer <HF_TOKEN>" \
  "http://35.184.58.132:7860/api/user/quota"
```

#### Response (200)
```json
{
  "plan": "org",
  "claude_used_today": 0,
  "claude_daily_cap": 999,
  "claude_remaining": 999
}
```

### POST `/api/interrupt/{session_id}`

#### curl
```bash
curl -s -X POST \
  -H "Authorization: Bearer <HF_TOKEN>" \
  "http://35.184.58.132:7860/api/interrupt/<session_id>"
```

#### Response (200)
```json
{
  "status": "interrupted",
  "session_id": "3b67f1cb-3ef4-4af4-b12c-1e0f3f2d9abc"
}
```

### GET `/api/session/{session_id}/messages`

#### curl
```bash
curl -s \
  -H "Authorization: Bearer <HF_TOKEN>" \
  "http://35.184.58.132:7860/api/session/<session_id>/messages"
```

#### Response (200)
```json
[
  {
    "role": "user",
    "content": "example",
    "tool_calls": null
  }
]
```

### POST `/api/undo/{session_id}`

#### curl
```bash
curl -s -X POST \
  -H "Authorization: Bearer <HF_TOKEN>" \
  "http://35.184.58.132:7860/api/undo/<session_id>"
```

#### Response (200)
```json
{
  "status": "undo_requested",
  "session_id": "3b67f1cb-3ef4-4af4-b12c-1e0f3f2d9abc"
}
```

### POST `/api/truncate/{session_id}`

#### curl
```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -d '{"user_message_index": 2}' \
  "http://35.184.58.132:7860/api/truncate/<session_id>"
```

#### Request JSON
```json
{
  "user_message_index": 2
}
```

#### Response (200)
```json
{
  "status": "truncated",
  "session_id": "3b67f1cb-3ef4-4af4-b12c-1e0f3f2d9abc"
}
```

### POST `/api/compact/{session_id}`

#### curl
```bash
curl -s -X POST \
  -H "Authorization: Bearer <HF_TOKEN>" \
  "http://35.184.58.132:7860/api/compact/<session_id>"
```

#### Response (200)
```json
{
  "status": "compact_requested",
  "session_id": "3b67f1cb-3ef4-4af4-b12c-1e0f3f2d9abc"
}
```

### POST `/api/shutdown/{session_id}`

#### curl
```bash
curl -s -X POST \
  -H "Authorization: Bearer <HF_TOKEN>" \
  "http://35.184.58.132:7860/api/shutdown/<session_id>"
```

#### Response (200)
```json
{
  "status": "shutdown_requested",
  "session_id": "3b67f1cb-3ef4-4af4-b12c-1e0f3f2d9abc"
}
```

---

## Skipped (as requested)

The following chat APIs are intentionally omitted:

- `POST /api/chat/{session_id}`
- `GET /api/events/{session_id}`
- `POST /api/platform/model-chat`
- `POST /api/submit`
- `POST /api/approve`
- `POST /api/title`
