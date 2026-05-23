# ML Intern Platform — End-to-End Implementation

## Goal
A single-container web service where users can:
1. Upload a dataset (CSV/JSON/JSONL) → pushed to HF Hub as a private dataset
2. Submit a fine-tuning job via the ml-intern agent → runs on HF compute → model pushed to HF Hub
3. Chat with the fine-tuned model via HF Inference API

Deploy target: **Google Cloud Run** (Docker container, port 7860)

---

## Current State

### What exists and works
- `backend/` — FastAPI app, session management, SSE streaming, tool approval flow
- `agent/` — autonomous agentic loop with 19 tools (hf_jobs, hf_repo_files, sandbox, etc.)
- `frontend/` — React + Vite app (being replaced with simple HTML)
- `Dockerfile` — multi-stage build (Node frontend + Python backend), HF Spaces compatible
- `backend/start.sh` — uvicorn entrypoint on port 7860
- Dev mode: set no `OAUTH_CLIENT_ID` → auth bypassed, all requests run as "dev" user
- HF token falls back to `HF_TOKEN` env var automatically (no user login needed)

### Key existing API endpoints (backend/routes/agent.py)
```
POST /api/session                      — create agent session
POST /api/chat/{session_id}            — submit text OR approvals + stream SSE events
GET  /api/events/{session_id}          — re-attach to running session SSE
POST /api/interrupt/{session_id}       — stop agent
GET  /api/health                       — health check
GET  /api/config/model                 — list available models
```

### SSE event types (from /api/chat/{session_id})
```
processing          — agent starting
assistant_chunk     — streaming token {"chunk": "..."}
assistant_message   — complete response {"content": "..."}
tool_call           — {"tool_name": "...", "tool_call_id": "...", "arguments": {...}}
tool_output         — {"tool_name": "...", "output": "..."}
approval_required   — {"tools": [{"tool": "...", "tool_call_id": "...", "arguments": {...}}]}
turn_complete       — turn finished
error               — {"message": "..."}
compacted           — context compressed
```

### Static file serving (backend/main.py)
```python
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
```
In Docker: `COPY --from=frontend-builder /app/frontend/dist ./static/`
→ We will replace this with `COPY backend/static/ ./static/`

---

## What to Build

### File changes overview
```
ADD  backend/routes/platform.py       — upload-dataset + model-chat routes
ADD  backend/static/index.html        — single-page HTML app (replaces React)
MOD  backend/main.py                  — include platform router
MOD  Dockerfile                       — remove Node build stage, copy static/ directly
ADD  .env                             — HF_TOKEN + optional ANTHROPIC_API_KEY
```

---

## Step 1: backend/routes/platform.py

Two new routes:

### POST /api/platform/upload-dataset
- Accepts multipart file upload + `repo_id` form field
- Creates HF dataset repo (private) if not exists
- Uploads file using `huggingface_hub.HfApi`
- Returns `{"dataset_id": "org/name", "filename": "data.csv"}`

```python
import os, tempfile
from fastapi import APIRouter, Depends, File, Form, UploadFile
from huggingface_hub import HfApi
from dependencies import get_current_user

router = APIRouter(prefix="/api/platform", tags=["platform"])

@router.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    repo_id: str = Form(...),
    user: dict = Depends(get_current_user),
):
    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=f"_{file.filename}", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=file.filename,
        repo_id=repo_id,
        repo_type="dataset",
    )
    os.unlink(tmp_path)
    return {"dataset_id": repo_id, "filename": file.filename}
```

### POST /api/platform/model-chat
- Accepts `{"model_id": "...", "messages": [...]}`
- Proxies to HF Router (`https://router.huggingface.co/v1/chat/completions`)
- Streams back SSE tokens
- Uses `HF_TOKEN` from env

```python
import httpx
from fastapi.responses import StreamingResponse

@router.post("/model-chat")
async def model_chat(body: dict, user: dict = Depends(get_current_user)):
    hf_token = os.environ.get("HF_TOKEN")
    model_id = body["model_id"]
    messages = body.get("messages", [])

    async def generate():
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                "https://router.huggingface.co/v1/chat/completions",
                headers={"Authorization": f"Bearer {hf_token}"},
                json={"model": model_id, "messages": messages, "stream": True, "max_tokens": 512},
            ) as resp:
                async for chunk in resp.aiter_text():
                    yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

---

## Step 2: backend/static/index.html

Single self-contained HTML file. No build step. Vanilla JS + embedded CSS.

### 3 screens (toggle visibility with JS)

#### Screen 1: Configure
```
┌──────────────────────────────────────────┐
│  HF Token (optional if set via env var)  │
│  Dataset: [Upload file] OR [HF URL]      │
│  Base Model: [google/gemma-2-2b-it]      │
│  Output Model Repo: [org/model-name]     │
│  Instructions: [textarea - pre-filled]   │
│  [▶ Start Fine-tuning]                   │
└──────────────────────────────────────────┘
```
Pre-filled instructions textarea:
```
Fine-tune the model on the provided dataset using TRL SFTTrainer with LoRA (r=16, alpha=32).
Use bf16, gradient checkpointing, 3 epochs, batch size 2, gradient accumulation 4.
Once training completes, push model and tokenizer to HF Hub as {output_repo}.
Submit as an HF Training Job on a T4 GPU.
```

#### Screen 2: Running (terminal output)
```
┌──────────────────────────────────────────┐
│  Status: ● Running  [■ Stop]             │
│  ─────────────────────────────────────   │
│  ▸ plan_tool  {"todos": [...]}           │
│    Plan updated:                         │
│      [~] 1. Inspect dataset...           │
│      [ ] 2. Write training script...     │
│                                          │
│  ▸ hf_inspect_dataset  {dataset...}      │
│    Rows: 50000, columns: [...]           │
│                                          │
│  ▸ hf_jobs  {operation: "run"...}        │
│    Downloading torch (506.1MiB)          │
│    Installed 88 packages                 │
│    Loading dataset...                    │
│    Starting training...                  │
│                                          │
│  ╔══════════════════════════════════╗    │
│  ║ Approval Required                ║    │
│  ║ hf_repo_files: upload train.py   ║    │
│  ║ [✓ Approve]  [✗ Deny]           ║    │
│  ╚══════════════════════════════════╝    │
└──────────────────────────────────────────┘
```

#### Screen 3: Chat
```
┌──────────────────────────────────────────┐
│  ✓ Model deployed: org/model-name        │
│  ─────────────────────────────────────   │
│  [Chat messages area]                    │
│                                          │
│  [Type a message...]          [Send →]   │
└──────────────────────────────────────────┘
```

### JS flow
```javascript
// 1. Upload file (if file selected)
const formData = new FormData();
formData.append('file', file);
formData.append('repo_id', 'org/dataset-name');
const { dataset_id } = await fetch('/api/platform/upload-dataset', {
  method: 'POST', body: formData
}).then(r => r.json());

// 2. Create agent session
const { session_id } = await fetch('/api/session', { method: 'POST' }).then(r => r.json());

// 3. Build prompt from form values and submit + stream
const prompt = buildPrompt(datasetId, baseModel, outputRepo, instructions);
const response = await fetch(`/api/chat/${session_id}`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({ text: prompt })
});

// 4. Read SSE stream
const reader = response.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split('\n');
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const event = JSON.parse(line.slice(6));
      handleEvent(event);
    }
  }
}

// 5. Handle approval_required
async function handleApproval(tools, approved) {
  const approvals = tools.map(t => ({
    tool_call_id: t.tool_call_id,
    approved,
    feedback: null
  }));
  const resp = await fetch(`/api/chat/${session_id}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ approvals })
  });
  // Resume reading SSE stream from resp
}

// 6. Chat with model (after turn_complete)
async function sendChatMessage(modelId, messages) {
  const resp = await fetch('/api/platform/model-chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ model_id: modelId, messages })
  });
  // Stream SSE tokens → append to chat
}
```

### Rendering events to terminal
```javascript
function handleEvent(event) {
  switch (event.event_type) {
    case 'tool_call':
      // Render: ▸ {tool_name}  {JSON.stringify(args, null, 0)}
      appendLine(`▸ ${event.tool_name}`, 'cyan');
      break;
    case 'tool_output':
      // Render indented, preserve newlines
      appendIndented(event.output, 'gray');
      break;
    case 'assistant_chunk':
      appendChunk(event.chunk, 'white');
      break;
    case 'assistant_message':
      appendLine(event.content, 'white');
      break;
    case 'approval_required':
      showApprovalPanel(event.tools);
      break;
    case 'turn_complete':
      showChatButton();
      break;
    case 'error':
      appendLine(`✗ Error: ${event.message}`, 'red');
      break;
  }
}
```

---

## Step 3: Update backend/main.py

Add one line to include the platform router:
```python
from routes.platform import router as platform_router
# ...
app.include_router(agent_router)
app.include_router(auth_router)
app.include_router(platform_router)   # ADD THIS
```

Change static path to serve from `backend/static/` (local dev):
```python
# Look for static files next to backend/ directory OR in backend/static/
static_path = Path(__file__).parent / "static"   # backend/static/
if not static_path.exists():
    static_path = Path(__file__).parent.parent / "static"   # /app/static/ in Docker
```

---

## Step 4: Simplify Dockerfile

Remove the Node.js build stage entirely. Copy `backend/static/` directly:

```dockerfile
# Single stage — no Node build needed
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN useradd -m -u 1000 user
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY agent/ ./agent/
COPY backend/ ./backend/
COPY configs/ ./configs/

# Static files — simple HTML, no build step
COPY backend/static/ ./static/

RUN mkdir -p /app/session_logs && chown -R user:user /app
USER user

ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 7860
WORKDIR /app/backend
CMD ["bash", "start.sh"]
```

---

## Step 5: Environment Variables

### Local .env (project root)
```
HF_TOKEN=hf_xxxx                    # HF write token — used by agent tools + inference
ANTHROPIC_API_KEY=sk-ant-xxxx       # Optional — makes agent smarter at writing scripts
# Leave OAUTH_CLIENT_ID unset → dev mode, no login required
```

### Cloud Run secrets (set via gcloud or console)
```
HF_TOKEN         → Secret Manager or plain env var
ANTHROPIC_API_KEY → Secret Manager (optional)
```
Do NOT set `OAUTH_CLIENT_ID` unless you want HF OAuth login.

---

## Step 6: Deploy to Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT/ml-intern .

# Deploy
gcloud run deploy ml-intern \
  --image gcr.io/YOUR_PROJECT/ml-intern \
  --port 7860 \
  --region us-central1 \
  --memory 2Gi \
  --set-env-vars HF_TOKEN="hf_xxxx" \
  --allow-unauthenticated
```

The service URL from Cloud Run is your platform URL.

---

## Full User Flow (end to end)

```
1. User opens https://your-cloud-run-url.run.app
   → Served by index.html (backend/static/index.html)

2. Screen 1: Configure
   - Optional: enter HF Token (or leave blank if HF_TOKEN is in env)
   - Upload CSV/JSON/JSONL file OR paste HF dataset URL
     → POST /api/platform/upload-dataset → file lands at org/dataset-name on HF Hub
   - Enter base model (e.g. google/gemma-2-2b-it)
   - Enter output model repo (e.g. ligaments-dev/my-finetuned-model)
   - Optionally edit instructions
   - Click "Start Fine-tuning"

3. Screen 2: Running
   - POST /api/session → get session_id
   - POST /api/chat/{session_id} with constructed prompt
   - SSE events stream in, rendered as terminal output
   - approval_required events show Approve/Deny buttons
     → user clicks Approve → POST /api/chat/{session_id} with approvals
   - Agent submits HF Training Job, monitors logs, pushes model
   - turn_complete → "Chat with Model" button appears

4. Screen 3: Chat
   - User types message
   - POST /api/platform/model-chat with {model_id, messages}
   - HF Router streams response tokens back
   - Rendered as chat bubbles
```

---

## Prompt Template (auto-constructed from form)

```python
def build_prompt(dataset_id, base_model, output_repo, extra_instructions):
    return f"""Dataset is {dataset_id} and model is {base_model}.
Perform full fine-tuning on this dataset using TRL SFTTrainer.
Use LoRA (r=16, alpha=32), bf16, gradient checkpointing, 3 epochs, batch size 2, gradient accumulation steps 4.
Once training completes, push the fine-tuned model and tokenizer to Hugging Face Hub as {output_repo}.
Submit the training job to HF Jobs on a T4 GPU.
{extra_instructions}"""
```

---

## Dependencies to add (pyproject.toml)

`huggingface_hub` is already a dependency. `httpx` is already a dependency.
No new packages needed.

---

## Testing Locally

```bash
# Install deps
uv sync

# Create backend/static/ directory
mkdir -p backend/static

# Run (no frontend build needed)
cd backend && uvicorn main:app --host 0.0.0.0 --port 7860 --reload

# Visit http://localhost:7860
```

---

## Notes & Gotchas

- **YOLO mode**: The agent will ask for tool approvals. The HTML page must handle `approval_required` events and send back approvals via SSE. Do not skip this.
- **SSE re-attachment**: After approvals, the `POST /api/chat/{session_id}` response IS the new SSE stream. Read from its body, not a separate GET.
- **HF Jobs timeout**: Training can take 30-60+ minutes. The SSE keepalive comment (`: keepalive\n\n`) fires every 15s to prevent proxy timeouts. Cloud Run has a max request timeout of 60 minutes — set `--timeout 3600` in gcloud deploy.
- **Model chat cold start**: First inference call on a new model via HF Router may take 30-60s to load. Show a loading indicator.
- **Static path in Docker**: `backend/main.py` looks for static files at `Path(__file__).parent.parent / "static"` which resolves to `/app/static/` in Docker. Keep `COPY backend/static/ ./static/` in Dockerfile.
- **Dev mode auth**: Without `OAUTH_CLIENT_ID` env var, all requests authenticate as user `"dev"` with `plan="org"`. This is intentional for Cloud Run deployment.
- **Token in env vs form**: If `HF_TOKEN` is set in Cloud Run env, the session creation in `backend/routes/agent.py` already picks it up as fallback (`os.environ.get("HF_TOKEN")`). No need to pass it from the frontend.
