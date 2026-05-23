# ML Intern - Demo_1 Autonomous Fine-Tuning

## Overview

`Demo_1` adds an autonomous fine-tuning demo platform to ML Intern. The branch is focused on a simple reviewer-friendly path: upload or mention a dataset, ask the agent to fine tune it, and receive a final Hugging Face model, job, and evaluation summary when the run completes.

The fine-tuning path is intentionally constrained for demo safety. It runs through Hugging Face Jobs only, does not require manual approval once the user asks for fine-tuning, keeps secrets on the backend, and applies a `$5` maximum budget guardrail before launching training.

## What This Branch Adds

- Frontend support for uploading or mentioning datasets, including `.pdf`, `.docx`, `.csv`, and `.xlsx` files.
- Backend dataset handling that creates Hugging Face dataset repositories under the authenticated HF username.
- Automatic strategy selection that decides whether the uploaded content is suitable for fine-tuning and chooses the right path.
- Hugging Face Jobs-only auto fine-tuning with no manual approval prompt on the fine-tune path.
- Stable template-based SFT training scripts with smoke/preflight validation before job launch.
- Deterministic script repair and classification so generated scripts do not drift into unsafe or unsupported behavior.
- Private backend-only handling for `HF_TOKEN` and `OPENAI_API_KEY`; `.env` must not be committed.
- Trackio warning/docs URLs filtered out of final model links, with Trackio disabled for the demo path.
- Final frontend display of model link, job link, and evaluation/result details.

Model quality depends on dataset size and quality. Small raw documents are useful for proving the pipeline, while strong task performance requires a larger instruction/QA-style dataset.

## Architecture/Key Files

- `agent/core/auto_finetune.py` orchestrates the automatic fine-tuning workflow.
- `agent/core/training_templates.py` provides stable SFT script templates used by Hugging Face Jobs.
- `agent/core/script_smoke.py` runs smoke/preflight validation before launching generated training scripts.
- `agent/core/dataset_inspection.py` inspects uploaded data and extracts useful dataset metadata.
- `agent/core/trainability.py` determines whether the dataset is suitable for fine-tuning.
- `agent/core/strategy_selector.py` chooses the execution strategy for the user request.
- `backend/services/dataset_service.py` uploads processed datasets to Hugging Face dataset repositories.
- `frontend` includes the `SessionFlow`, `useAgentChat`, and `autoFineTuneResult` UI paths that surface upload state and final fine-tuning results.

## Quick Start

### Backend - Windows CMD

```cmd
cd /d D:\_AI_\LigaX_ML_Intern\huggingface-ml-intern-finetuning
git switch Demo_1
uv sync --extra dev
set PYTHONPATH=backend
uv run uvicorn main:app --host 0.0.0.0 --port 7860 --app-dir backend
```

### Frontend - Windows CMD

```cmd
cd /d D:\_AI_\LigaX_ML_Intern\huggingface-ml-intern-finetuning\frontend
npm install
npm run dev
```

Open `http://localhost:5173/`.

### Required Environment

Create a local `.env` file in the project root. Do not commit it.

```env
HF_TOKEN=<your-hugging-face-token>
OPENAI_API_KEY=<your-openai-api-key>
```

`HF_TOKEN` is used for Hugging Face dataset repositories, model repositories, and Jobs. `OPENAI_API_KEY` is backend-only and is used for repair when configured.

## End-to-End Demo Flow

1. Start the backend and frontend.
2. Open `http://localhost:5173/`.
3. Upload a supported file (`.pdf`, `.docx`, `.csv`, or `.xlsx`) or mention an existing dataset.
4. Send:

```text
fine tune this dataset
```

5. Wait for the backend to inspect the dataset, choose the strategy, create a Hugging Face dataset repository, validate the training script, and launch the Hugging Face Job.
6. Review the final frontend result containing the model link, job link, and evaluation/result details.

Example successful demo artifacts:

- Model: https://huggingface.co/ligaments-dev/generic-session-1c75f299-auto-sft
- Job: https://huggingface.co/jobs/ligaments-dev/6a11bc6be3c0b51e1ca5de32

## Verification

Recent verification for this branch:

- Frontend tests passed.
- Frontend lint passed.
- Frontend build passed.
- `ruff format` and `ruff check` passed.
- CI on PR #1 passed after formatting.

For README-only changes, review the markdown diff before pushing:

```cmd
git diff -- README.md
```

## Known Limitations

- Small raw PDF datasets can prove the pipeline but usually will not produce strong model quality.
- High-quality answers require a larger instruction/QA dataset with enough examples for SFT.
- Fine-tuning currently runs through Hugging Face Jobs only.
- Other cloud providers are dry-run or disabled in this branch.

## Review/Merge Notes

- `Demo_1` is separate from `main`.
- No merge conflicts were shown at handoff time.
- Checks were green at handoff time.
- The fine-tuning demo keeps secrets backend-only and should not require committing `.env` or other local credentials.
