"""Launch a capped Hugging Face Jobs SFT run for the Income Tax QA dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi

from agent.core.cost_estimation import estimate_hf_job_cost
from agent.tools.jobs_tool import HfJobsTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--hardware", default="t4-small")
    parser.add_argument("--timeout", default="2h")
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=20)
    return parser.parse_args()


def build_training_script(args: argparse.Namespace) -> str:
    return f'''
# /// script
# dependencies = [
#   "accelerate>=1.0.0",
#   "datasets>=3.0.0",
#   "huggingface-hub>=0.27.0",
#   "peft>=0.13.0",
#   "trackio>=0.4.0",
#   "transformers>=4.46.0",
#   "trl>=0.12.0",
# ]
# ///

import os

import trackio
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

dataset_repo = "{args.dataset_repo}"
model_repo = "{args.model_repo}"
base_model = "{args.base_model}"
max_steps = {args.max_steps}
PROJECT = "income-tax-master-qa"
RUN_NAME = "qwen-0.5b-lora-demo"

token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN is required for private dataset/model access")

api = HfApi(token=token)
api.create_repo(repo_id=model_repo, repo_type="model", private=True, exist_ok=True)

dataset = load_dataset(dataset_repo, split="train", token=token)
print(f"Loaded {{len(dataset)}} Income Tax QA examples from {{dataset_repo}}")

tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    trackio.init(project=PROJECT, name=RUN_NAME)
except Exception as exc:
    print(f"Trackio initialization failed: {{exc}}")

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

training_args = SFTConfig(
    output_dir="income-tax-master-qa-assistant",
    max_steps=max_steps,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=2,
    logging_steps=1,
    save_steps=max_steps,
    save_total_limit=1,
    max_length=512,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    push_to_hub=True,
    hub_model_id=model_repo,
    hub_private_repo=True,
    report_to=["trackio"],
    run_name=RUN_NAME,
)

trainer = SFTTrainer(
    model=base_model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.train()
trainer.push_to_hub()
print(f"Pushed private LoRA adapter model to https://huggingface.co/{{model_repo}}")
'''


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not configured")

    estimate = await estimate_hf_job_cost(
        {"operation": "run", "hardware_flavor": args.hardware, "timeout": args.timeout}
    )
    if estimate.estimated_cost_usd is None:
        raise RuntimeError(estimate.block_reason or "Could not estimate HF Jobs cost")
    if estimate.estimated_cost_usd > args.budget_usd:
        raise RuntimeError(
            f"Estimated cost ${estimate.estimated_cost_usd:.2f} exceeds "
            f"budget cap ${args.budget_usd:.2f}"
        )

    whoami = HfApi(token=token).whoami()
    namespace = whoami["name"]
    print(
        json.dumps(
            {
                "dataset_repo": args.dataset_repo,
                "model_repo": args.model_repo,
                "base_model": args.base_model,
                "hardware": args.hardware,
                "timeout": args.timeout,
                "estimated_cost_usd": estimate.estimated_cost_usd,
            },
            indent=2,
        )
    )

    tool = HfJobsTool(hf_token=token, namespace=namespace)
    result = await tool.execute(
        {
            "operation": "run",
            "script": build_training_script(args),
            "dependencies": [
                "accelerate>=1.0.0",
                "datasets>=3.0.0",
                "huggingface-hub>=0.27.0",
                "peft>=0.13.0",
                "trackio>=0.4.0",
                "transformers>=4.46.0",
                "trl>=0.12.0",
            ],
            "hardware_flavor": args.hardware,
            "timeout": args.timeout,
        }
    )
    print(result["formatted"])


if __name__ == "__main__":
    asyncio.run(main())
