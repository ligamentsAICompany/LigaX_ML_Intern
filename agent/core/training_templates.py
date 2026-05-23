"""Stable template-based training scripts for automatic fine-tuning."""

from __future__ import annotations

import json


def _py(value: object) -> str:
    """Render a Python literal from JSON-safe data."""

    return json.dumps(value)


def render_sft_training_script(
    *,
    dataset_repo: str,
    output_model_repo: str,
    base_model: str,
    max_length: int,
    trackio_project: str,
    trackio_run_name: str,
) -> str:
    """Render a current-compatible TRL SFT script.

    The template deliberately keeps all core APIs literal so static smoke checks
    can validate them before HF Jobs spend starts.
    """

    return f'''# /// script
# dependencies = ["accelerate", "datasets", "huggingface-hub", "peft", "torch", "transformers", "trl"]
# ///

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


DATASET_REPO = {_py(dataset_repo)}
BASE_MODEL = {_py(base_model)}
OUTPUT_MODEL_REPO = {_py(output_model_repo)}


def _load_split():
    dataset = load_dataset(DATASET_REPO)
    if "train" in dataset:
        train = dataset["train"]
    else:
        first_split = next(iter(dataset))
        train = dataset[first_split]
    if "validation" in dataset:
        return train, dataset["validation"]
    if len(train) >= 20:
        split = train.train_test_split(test_size=0.1, seed=42)
        return split["train"], split["test"]
    return train, None


def _format_example(example):
    if isinstance(example.get("messages"), list):
        return {{"messages": example["messages"]}}
    prompt = (
        example.get("prompt")
        or example.get("question")
        or example.get("instruction")
        or example.get("input")
        or example.get("text")
        or ""
    )
    response = (
        example.get("response")
        or example.get("answer")
        or example.get("output")
        or example.get("completion")
        or ""
    )
    if response:
        text = f"### User:\\n{{prompt}}\\n\\n### Assistant:\\n{{response}}"
    else:
        text = str(prompt)
    return {{"text": text}}


def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for pushing the fine-tuned model.")

    train_dataset, eval_dataset = _load_split()
    train_dataset = train_dataset.map(_format_example, remove_columns=train_dataset.column_names)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(_format_example, remove_columns=eval_dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, token=token)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        output_dir="outputs/auto-sft",
        hub_model_id="{output_model_repo}",
        push_to_hub=True,
        max_length={int(max_length)},
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=25,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate() if eval_dataset is not None else {{}}
    trainer.push_to_hub()

    Path("auto_finetune_result.json").write_text(
        json.dumps(
            {{
                "model_repo": OUTPUT_MODEL_REPO,
                "model_url": f"https://huggingface.co/{{OUTPUT_MODEL_REPO}}",
                "eval": metrics,
            }},
            indent=2,
            sort_keys=True,
        )
    )
    print("AUTO_FINETUNE_MODEL_URL=" + f"https://huggingface.co/{{OUTPUT_MODEL_REPO}}")
    print("AUTO_FINETUNE_EVAL=" + json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
'''
