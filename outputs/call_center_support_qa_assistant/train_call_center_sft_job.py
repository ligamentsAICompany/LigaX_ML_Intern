# /// script
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

DATASET_REPO = "ligaments-dev/call-center-support-qa-assistant-option4-data"
MODEL_REPO = "ligaments-dev/call-center-support-qa-assistant-option4-lora"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TRAIN_ROWS = 1200
MAX_EVAL_ROWS = 100
PROJECT = "call-center-support-qa-option4"
RUN_NAME = "qwen05b-lora-smoke"
TRAINABILITY = {
    "score": 90,
    "recommendation": "fine_tune",
    "risk_level": "low",
    "reasons": [
        "Dataset has instruction/SFT-style fields suitable for supervised fine-tuning.",
        "Dataset format (csv) is tabular and needs task conversion before SFT.",
    ],
}
STRATEGY = {
    "strategy": "fine_tune",
    "confidence": 0.9,
    "risk_level": "low",
    "reasons": [
        "Recommended method: supervised fine-tuning.",
        "Dataset has valid instruction/SFT-style fields for training.",
        "Dataset has instruction/SFT-style fields suitable for supervised fine-tuning.",
        "Dataset format (csv) is tabular and needs task conversion before SFT.",
    ],
    "required_next_actions": [
        "Run a small SFT job with held-out evaluation before scaling.",
        "Track quality and overfitting metrics during training.",
    ],
    "can_train_without_override": True,
    "requires_user_override_for_training": False,
    "method_hint": "sft",
    "override_message": "",
    "metadata": {
        "trainability": {
            "score": 90,
            "recommendation": "fine_tune",
            "risk_level": "low",
            "reasons": [
                "Dataset has instruction/SFT-style fields suitable for supervised fine-tuning.",
                "Dataset format (csv) is tabular and needs task conversion before SFT.",
            ],
        },
        "method_recommendation": "sft",
    },
}
GOLDEN_EVAL = {
    "case_count": 25,
    "quality_constraints": [
        "Do not hallucinate values",
        "Expected answer must be concise",
        "Answer must be supported by source fields",
    ],
}


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN secret is required for private dataset/model access."
        )

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
            print(f"Trackio initialization warning: {exc}")

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

    report = {
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
    }
    Path("post_training_eval.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
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
