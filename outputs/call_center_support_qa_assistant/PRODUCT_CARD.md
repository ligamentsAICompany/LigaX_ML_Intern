# Call Center Support QA Assistant

Status: ready for pilot validation; private LoRA adapter trained and local grounded reference assistant available.

Use `assistant_product.py` for local retrieval-style answers while the private LoRA job is evaluated.

# Model Provenance

## Source
- Base model: Qwen/Qwen2.5-0.5B-Instruct
- Dataset: D:\_AI_\LigaX_ML_Intern\huggingface-ml-intern-finetuning\bitext-telco-llm-chatbot-training-dataset.csv
- Row count: 26000

## Training
- Method: sft_lora
- Hardware: {'provider': 'hf-jobs', 'flavor': 't4-small'}
- Timeout: 2h

## Quality Intelligence
- Trainability Risk: low
- Strategy Recommendation: fine_tune
- Golden Eval: 25 cases
- Post-Training Eval: passed

## Limitations
- Local reference assistant passed deterministic source-grounded eval; trained model eval is pending until the HF Job finishes.
- Dataset is public support-template data with placeholders, not private company policy transcripts.
- Human QA review is required before production use.