"""
Full fine-tuning (QLoRA 4-bit NF4) of Google Gemma-2B-IT on the Bitext telco chatbot dataset.
Deploys the trained model to Hugging Face Hub.

Note: True 2-bit training is not supported by standard libraries (bitsandbytes only supports 4-bit/8-bit).
We use 4-bit NF4 (NormalFloat4) which is the industry-standard memory-efficient quantization approach.
This provides ~4x memory savings compared to FP16, enabling fine-tuning on consumer GPUs.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import trackio

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-2b-it"
DATASET_ID = "bitext/Bitext-telco-llm-chatbot-training-dataset"
HUB_MODEL_ID = "ligaments-dev/gemma-2b-telco-sft"
OUTPUT_DIR = "./gemma-telco-sft-output"

# ── Initialize Trackio for monitoring ──────────────────────────────────────
trackio.init(
    project="gemma-telco-sft",
    name="gemma-2b-telco-qlora-4bit",
    config={
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "quantization": "4bit-nf4",
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": 3,
        "learning_rate": 2e-4,
    },
)

# ── 1. Load & format dataset ───────────────────────────────────────────────
print("Loading dataset...")
dataset = load_dataset(DATASET_ID, split="train")
print(f"Dataset loaded: {len(dataset)} examples")


def format_to_messages(example):
    """Convert instruction/response to conversational messages format."""
    return {
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }


dataset = dataset.map(format_to_messages, remove_columns=dataset.column_names)
print(f"Formatted dataset sample: {dataset[0]}")

# ── 2. Load tokenizer ──────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ── 3. Quantization config (4-bit NF4 — closest practical to 2-bit) ────────
print("Setting up 4-bit NF4 quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4 — optimal for weight distributions
    bnb_4bit_use_double_quant=True,  # Nested quantization saves more memory
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16 for stability
)

# ── 4. Load model with quantization ──────────────────────────────────────────
print("Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.config.use_cache = False  # Required for gradient checkpointing
print("Model loaded. Trainable params info will be shown after LoRA setup.")

# ── 5. LoRA config (PEFT adapters for efficient fine-tuning) ─────────────────
print("Applying LoRA adapters...")
peft_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # Scaling factor
    target_modules="all-linear",  # Auto-detect all linear layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ── 6. Training config ───────────────────────────────────────────────────────
print("Configuring training arguments...")
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="paged_adamw_8bit",  # Paged optimizer for memory efficiency
    bf16=True,
    gradient_checkpointing=True,  # Trade compute for memory
    logging_strategy="steps",
    logging_steps=10,
    logging_first_step=True,
    save_strategy="epoch",
    save_total_limit=2,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    hub_private_repo=False,
    report_to=["trackio"],
    max_length=512,
    packing=False,
    disable_tqdm=True,
    seed=42,
)

# ── 7. Initialize trainer ──────────────────────────────────────────────────
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

# ── 8. Train ─────────────────────────────────────────────────────────────────
print("Starting training...")
trainer.train()

# ── 9. Save & deploy ───────────────────────────────────────────────────────────
print("Saving final model...")
trainer.save_model(OUTPUT_DIR)

print("Pushing to Hugging Face Hub...")
trainer.push_to_hub(
    commit_message="Fine-tuned Gemma-2B-IT on Bitext telco chatbot dataset (QLoRA 4-bit NF4)"
)

print("Training complete! Model deployed to:")
print(f"  https://huggingface.co/{HUB_MODEL_ID}")

# ── 10. Merge adapters for inference (optional but recommended) ──────────────
print("Merging LoRA adapters with base model for optimized inference...")
merged_model = model.merge_and_unload()
merged_model.push_to_hub(
    f"{HUB_MODEL_ID}-merged",
    commit_message="Merged Gemma-2B-IT + LoRA adapters for inference",
)
print(f"Merged model deployed to: https://huggingface.co/{HUB_MODEL_ID}-merged")
