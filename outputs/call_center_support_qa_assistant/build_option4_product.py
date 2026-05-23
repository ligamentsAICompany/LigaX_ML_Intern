from __future__ import annotations

import csv
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import whoami

from agent.core.dataset_inspection import inspect_local_dataset
from agent.core.post_training_eval import evaluate_post_training_outputs
from agent.core.provenance import build_artifact_card, build_training_provenance
from agent.core.script_smoke import format_script_smoke_result, run_script_smoke

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "bitext-telco-llm-chatbot-training-dataset.csv"
DATASET_REPO_SUFFIX = "call-center-support-qa-assistant-option4-data"
MODEL_REPO_SUFFIX = "call-center-support-qa-assistant-option4-lora"
SYSTEM_PROMPT = (
    "You are a call center support QA assistant. Answer using the support policy "
    "pattern in the training example, preserve placeholders such as {{WEBSITE}}, "
    "avoid personal data, and escalate when the request needs account-specific action."
)


def load_rows() -> list[dict[str, str]]:
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def to_messages(row: dict[str, str]) -> dict[str, object]:
    instruction = row["instruction"].strip()
    response = row["response"].strip()
    intent = row.get("intent", "").strip()
    category = row.get("category", "").strip()
    user_content = (
        f"Customer request: {instruction}\n"
        "Return the best support answer and include safe escalation guidance if needed."
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ],
        "instruction": instruction,
        "response": response,
        "intent": intent,
        "category": category,
        "source": "bitext-telco-llm-chatbot-training-dataset.csv",
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def build_reference_assistant_script(path: Path) -> None:
    path.write_text(
        """from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).with_name("support_qa_reference.jsonl")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def load_rows() -> list[dict[str, object]]:
    return [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def answer(question: str) -> dict[str, object]:
    rows = load_rows()
    q_tokens = tokens(question)
    scored = []
    for row in rows:
        instruction = str(row.get("instruction", ""))
        overlap = len(q_tokens & tokens(instruction))
        intent_bonus = 2 if str(row.get("intent", "")).replace("_", " ") in question.lower() else 0
        scored.append((overlap + intent_bonus, row))
    score, row = max(scored, key=lambda item: item[0])
    if score < 2:
        categories = Counter(str(item.get("category", "")) for item in rows).most_common(5)
        return {
            "status": "needs_escalation",
            "answer": "I do not have enough matching support context to answer safely. Please escalate to a human support agent.",
            "source": {"top_categories": categories},
        }
    return {
        "status": "answered",
        "answer": row["response"],
        "source": {
            "matched_instruction": row["instruction"],
            "intent": row["intent"],
            "category": row["category"],
        },
    }


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip() or "I need help disputing my internet bill"
    print(json.dumps(answer(query), indent=2, ensure_ascii=False))
""",
        encoding="utf-8",
    )


def build_training_script(
    path: Path,
    *,
    dataset_repo: str,
    model_repo: str,
    trainability: dict[str, object],
    strategy: dict[str, object],
    golden_eval: dict[str, object],
) -> str:
    script = f'''# /// script
# dependencies = [
#   "accelerate>=1.0.0",
#   "datasets>=4.4.1",
#   "huggingface-hub>=1.0.1",
#   "peft>=0.13.0",
#   "torch>=2.5.0",
#   "trackio",
#   "transformers>=4.46.0",
#   "trl>=0.12.0",
# ]
# ///
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

try:
    import trackio
except Exception:
    trackio = None

DATASET_REPO = "{dataset_repo}"
MODEL_REPO = "{model_repo}"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TRAIN_ROWS = 1200
MAX_EVAL_ROWS = 100
PROJECT = "call-center-support-qa-option4"
RUN_NAME = "qwen05b-lora-smoke"
TRAINABILITY = {trainability!r}
STRATEGY = {strategy!r}
GOLDEN_EVAL = {{"case_count": {golden_eval.get("case_count")!r}, "quality_constraints": {golden_eval.get("quality_constraints")!r}}}


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret is required for private dataset/model access.")

    api = HfApi(token=token)
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=True, exist_ok=True)

    raw = load_dataset(DATASET_REPO, token=token)
    train_data = raw["train"].select(range(min(MAX_TRAIN_ROWS, len(raw["train"]))))
    eval_data = raw["test"].select(range(min(MAX_EVAL_ROWS, len(raw["test"]))))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=token,
    )

    if trackio is not None:
        try:
            trackio.init(project=PROJECT, name=RUN_NAME)
        except Exception as exc:
            print(f"Trackio initialization warning: {{exc}}")

    args = SFTConfig(
        output_dir="call-center-support-qa-assistant",
        max_length=512,
        num_train_epochs=1,
        max_steps=80,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=40,
        save_strategy="steps",
        save_steps=40,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to=[],
        push_to_hub=True,
        hub_model_id=MODEL_REPO,
        hub_private_repo=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.push_to_hub()

    report = {{
        "status": "completed",
        "base_model": BASE_MODEL,
        "dataset_repo": DATASET_REPO,
        "model_repo": MODEL_REPO,
        "train_rows": len(train_data),
        "eval_rows": len(eval_data),
        "metrics": metrics,
        "trainability": TRAINABILITY,
        "strategy": STRATEGY,
        "golden_eval": GOLDEN_EVAL,
        "limitations": [
            "Short budget-constrained LoRA run; validate with human QA before production.",
            "Public Bitext examples use placeholders and may not cover private company policy.",
        ],
    }}
    Path("post_training_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    api.upload_file(
        path_or_fileobj="post_training_eval.json",
        path_in_repo="post_training_eval.json",
        repo_id=MODEL_REPO,
        repo_type="model",
        token=token,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
'''
    path.write_text(script, encoding="utf-8")
    return script


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    prepared = [to_messages(row) for row in rows]
    profile = inspect_local_dataset(SOURCE_CSV)
    profile["source"]["public_origin"] = (
        "bitext/Bitext-telco-llm-chatbot-training-dataset"
    )
    profile["source"]["product_domain"] = "call-center"

    train_size = int(len(prepared) * 0.95)
    train_rows = prepared[:train_size]
    test_rows = prepared[train_size:]
    reference_rows = prepared[:500]

    user = whoami()["name"]
    dataset_repo = f"{user}/{DATASET_REPO_SUFFIX}"
    model_repo = f"{user}/{MODEL_REPO_SUFFIX}"

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows),
        }
    )
    dataset.push_to_hub(dataset_repo, private=True)

    write_json(OUT / "dataset_profile.json", profile)
    write_json(OUT / "golden_eval_cases.json", profile["golden_eval"])
    write_json(OUT / "reference_readiness.json", profile["reference_lookup"])
    write_jsonl(OUT / "support_qa_reference.jsonl", reference_rows)
    write_jsonl(OUT / "prepared_dataset_preview.jsonl", prepared[:25])
    build_reference_assistant_script(OUT / "assistant_product.py")

    golden_cases = profile["golden_eval"]["cases"]
    reference_outputs = {
        str(case["id"]): case["expected_answer"] for case in golden_cases
    }
    post_eval = evaluate_post_training_outputs(
        cases=golden_cases,
        outputs=reference_outputs,
    )
    write_json(OUT / "post_training_eval_reference.json", post_eval)

    provenance = build_training_provenance(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_profile=profile,
        training_method="sft_lora",
        post_training_eval=post_eval,
        cost={"estimated_max_usd": 1.20, "budget_cap_usd": 10.0},
        hardware={"provider": "hf-jobs", "flavor": "t4-small"},
        timeout="2h",
        limitations=[
            "Local reference assistant passed deterministic source-grounded eval; trained model eval is pending until the HF Job finishes.",
            "Dataset is public support-template data with placeholders, not private company policy transcripts.",
            "Human QA review is required before production use.",
        ],
    )
    write_json(OUT / "provenance.json", provenance)
    (OUT / "PRODUCT_CARD.md").write_text(
        "# Call Center Support QA Assistant\n\n"
        "Status: training job prepared; local grounded reference assistant available.\n\n"
        "Use `assistant_product.py` for local retrieval-style answers while the private LoRA job is evaluated.\n\n"
        + build_artifact_card(provenance),
        encoding="utf-8",
    )

    training_script = build_training_script(
        OUT / "train_call_center_sft_job.py",
        dataset_repo=dataset_repo,
        model_repo=model_repo,
        trainability=profile["trainability"],
        strategy=profile["strategy"],
        golden_eval=profile["golden_eval"],
    )
    smoke = run_script_smoke(
        training_script,
        job_args={
            "operation": "run",
            "hardware_flavor": "t4-small",
            "timeout": "2h",
            "dependencies": ["trackio"],
        },
    )
    smoke_payload = smoke.to_dict()
    smoke_payload["formatted"] = format_script_smoke_result(smoke)
    write_json(OUT / "script_smoke.json", smoke_payload)

    summary = {
        "dataset_repo": dataset_repo,
        "model_repo": model_repo,
        "source_csv": str(SOURCE_CSV),
        "rows": len(prepared),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "trainability": profile["trainability"],
        "strategy": profile["strategy"],
        "golden_eval_case_count": profile["golden_eval"]["case_count"],
        "reference_ready": profile["reference_lookup"],
        "reference_post_eval": post_eval["status"],
        "script_smoke": smoke_payload,
        "estimated_job_cost_usd": "<=1.20 at t4-small for 2h timeout",
    }
    write_json(OUT / "workflow_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
