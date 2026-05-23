"""
Full fine-tuning script:
  Model: google/gemma-2-2b-it
  Dataset: talkmap/telecom-conversation-corpus
  Converts turn-based telecom dialogues → conversational messages format for SFT.
"""

from collections import defaultdict

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import trackio
import torch

# ─── Config ─────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-2-2b-it"
DATASET_ID = "talkmap/telecom-conversation-corpus"
OUTPUT_DIR = "./gemma-2b-it-telecom"
HUB_MODEL_ID = "ligaments-dev/gemma-2b-it-telecom"
MAX_SEQ_LENGTH = 2048

# ─── Trackio ───────────────────────────────────────────────────────
trackio.init(
    project="gemma-telecom-finetune",
    name="gemma-2b-it-full-sft",
)

# ─── Load dataset ──────────────────────────────────────────────────
print("Loading dataset...")
ds = load_dataset(DATASET_ID, split="train")
print(f"Rows: {len(ds)}, Columns: {ds.column_names}")

# Group rows by conversation_id and sort by date_time to reconstruct dialogues
print("Grouping conversations...")
conversations = defaultdict(list)
for row in ds:
    conversations[row["conversation_id"]].append(row)

for conv_id in conversations:
    conversations[conv_id].sort(key=lambda x: x["date_time"])

# Convert each conversation into messages format
print("Converting to messages format...")
messages_data = []
for conv_id, turns in conversations.items():
    messages = []
    for turn in turns:
        role = "user" if turn["speaker"] == "client" else "assistant"
        messages.append({"role": role, "content": turn["text"]})
    messages_data.append({"messages": messages})

train_dataset = Dataset.from_list(messages_data)
print(f"Total conversations: {len(train_dataset)}")

# ─── Tokenizer ─────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ─── Model ─────────────────────────────────────────────────────────
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# ─── Training arguments ──────────────────────────────────────────────
args = SFTConfig(
    output_dir=OUTPUT_DIR,
    hub_model_id=HUB_MODEL_ID,
    push_to_hub=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_length=MAX_SEQ_LENGTH,
    logging_strategy="steps",
    logging_steps=10,
    logging_first_step=True,
    disable_tqdm=True,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    report_to=["trackio"],
    remove_unused_columns=False,
)

# ─── Trainer ───────────────────────────────────────────────────────
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

# ─── Train ─────────────────────────────────────────────────────────
print("Starting training...")
trainer.train()

# ─── Save & Push ───────────────────────────────────────────────────
print("Saving and pushing to hub...")
trainer.save_model(OUTPUT_DIR)
trainer.push_to_hub()

print(f"✅ Model pushed to https://huggingface.co/{HUB_MODEL_ID}")
