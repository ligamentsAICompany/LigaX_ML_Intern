# All APIs (Complete Reference)

Base URL:

`http://35.184.58.132:7860`

Docs:

- Swagger: `http://35.184.58.132:7860/docs`
- ReDoc: `http://35.184.58.132:7860/redoc`

---

## A) System

- `GET /api`
- `GET /api/health`
- `GET /api/health/llm`

Example:
```bash
curl -s "http://35.184.58.132:7860/api/health"
```

---

## B) Auth

- `GET /auth/status`
- `GET /auth/login`
- `GET /auth/callback` (OAuth redirect target)
- `GET /auth/logout`
- `GET /auth/me`
- `GET /auth/org-membership`

Example:
```bash
curl -s -H "Authorization: Bearer <HF_TOKEN>" "http://35.184.58.132:7860/auth/me"
```

---

## C) Platform

### Dataset upload
- `POST /api/platform/upload-dataset`

```bash
curl -s \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -F "file=@bitext-telco-llm-chatbot-training-dataset.csv" \
  -F "repo_id=ligaments-dev/Qwen-telecom-chatbot-data" \
  "http://35.184.58.132:7860/api/platform/upload-dataset"
```

### Model chat (SSE)
- `POST /api/platform/model-chat`

```bash
curl -N \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"ligaments-dev/Qwen-telecom-chatbot-model","messages":[{"role":"user","content":"Hi"}]}' \
  "http://35.184.58.132:7860/api/platform/model-chat"
```

---

## D) Agent Config + Metadata

- `GET /api/config/model`
- `GET /api/user/quota`
- `GET /api/sessions`
- `GET /api/session/{session_id}`
- `GET /api/session/{session_id}/messages`

---

## E) Session Lifecycle

- `POST /api/session`
- `POST /api/session/restore-summary`
- `POST /api/session/{session_id}/model`
- `DELETE /api/session/{session_id}`

Create session:
```bash
curl -s \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"model":"moonshotai/Kimi-K2.6"}' \
  "http://35.184.58.132:7860/api/session"
```

---

## F) Core Chat/Execution APIs (Agent SSE)

- `POST /api/chat/{session_id}` (submit text or approvals and stream events)
- `GET /api/events/{session_id}` (reattach stream)
- `POST /api/submit`
- `POST /api/approve`
- `POST /api/title`

SSE run example:
```bash
curl -N \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"text":"Train model using dataset ligaments-dev/Qwen-telecom-chatbot-data"}' \
  "http://35.184.58.132:7860/api/chat/<session_id>"
```

Approval example:
```bash
curl -s \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<session_id>","approvals":[{"tool_call_id":"call_123","approved":true}]}' \
  "http://35.184.58.132:7860/api/approve"
```

---

## G) Session Controls

- `POST /api/interrupt/{session_id}`
- `POST /api/undo/{session_id}`
- `POST /api/truncate/{session_id}`
- `POST /api/compact/{session_id}`
- `POST /api/shutdown/{session_id}`

Interrupt example:
```bash
curl -s -X POST \
  -H "Authorization: Bearer <HF_TOKEN>" \
  "http://35.184.58.132:7860/api/interrupt/<session_id>"
```

Truncate example:
```bash
curl -s -X POST \
  -H "Authorization: Bearer <HF_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"user_message_index":2}' \
  "http://35.184.58.132:7860/api/truncate/<session_id>"
```

---

## Auth Header/Cookie behavior

Token resolution order on backend:

1. `Authorization: Bearer <token>`
2. `hf_access_token` cookie
3. server `HF_TOKEN` env fallback

For browser calls using OAuth cookie:

```js
fetch(url, { credentials: "include" })
```

