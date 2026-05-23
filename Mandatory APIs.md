# Mandatory APIs (UI Integration)

Base URL:

`http://35.184.58.132:7860`

These are the minimum endpoints needed for your current flow (upload dataset + run fine-tune agent workflow).

---

## 1) API Root (sanity check)

### GET `/api`

```bash
curl -s "http://35.184.58.132:7860/api"
```

Response:
```json
{
  "name": "HF Agent API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

---

## 2) Upload Dataset

### POST `/api/platform/upload-dataset`

```bash
curl -s \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -F "file=@bitext-telco-llm-chatbot-training-dataset.csv" \
  -F "repo_id=ligaments-dev/Qwen-telecom-chatbot-data" \
  "http://35.184.58.132:7860/api/platform/upload-dataset"
```

Request (multipart/form-data):
- `file` (required)
- `repo_id` (required, `org/name`)

Response:
```json
{
  "dataset_id": "ligaments-dev/Qwen-telecom-chatbot-data",
  "filename": "bitext-telco-llm-chatbot-training-dataset.csv",
  "url": "https://huggingface.co/datasets/ligaments-dev/Qwen-telecom-chatbot-data"
}
```

---

## 3) Create Agent Session

### POST `/api/session`

```bash
curl -s \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -d '{"model":"moonshotai/Kimi-K2.6"}' \
  "http://35.184.58.132:7860/api/session"
```

Response:
```json
{
  "session_id": "3b67f1cb-3ef4-4af4-b12c-1e0f3f2d9abc",
  "ready": true
}
```

---

## 4) Run/Stream Agent Turn (SSE)

### POST `/api/chat/{session_id}`

```bash
curl -N \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -d '{"text":"Dataset is ... Submit as an HF Training Job on a T4 GPU."}' \
  "http://35.184.58.132:7860/api/chat/<session_id>"
```

Request JSON (one of):

```json
{ "text": "..." }
```

or approval continuation:

```json
{
  "approvals": [
    { "tool_call_id": "call_123", "approved": true, "feedback": null }
  ]
}
```

Response:
- `text/event-stream`
- `data: {...}` events until `turn_complete` / `approval_required` / `error`

---

## 5) Interrupt Running Turn

### POST `/api/interrupt/{session_id}`

```bash
curl -s -X POST \
  -H "Authorization: Bearer <HF_TOKEN>" \
  "http://35.184.58.132:7860/api/interrupt/<session_id>"
```

Response:
```json
{
  "status": "interrupted",
  "session_id": "3b67f1cb-3ef4-4af4-b12c-1e0f3f2d9abc"
}
```

---

## Browser note

If using cookie auth (`/auth/login`) from UI, send:

```js
fetch(url, { credentials: "include" })
```

