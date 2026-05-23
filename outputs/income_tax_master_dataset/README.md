---
license: other
task_categories:
- text-generation
language:
- en
tags:
- income-tax
- structured-reference
pretty_name: Income Tax Master
---

# Income Tax Master

Converted from a local Excel workbook into JSONL for Dataset Viewer and the local auto fine-tune/RAG pipeline.

## Data Files

- `data/train.jsonl`: one workbook row per record with `sheet`, `row_number`, `text`, and nested `record` fields.

## Schema Summary

- Income Tax Master: 26 rows; columns: Form_No, Category, Purpose, Applicable_Sections, Filing_Frequency, Due_Dates, User_Categories, Old_New_Regime_Applicability, API___E_Filing_Mapping_Possibilities

## Recommended Use

This appears to be a structured reference workbook, so retrieval/reference lookup is likely safer than supervised fine-tuning unless question-answer or instruction-response pairs are later curated from it.
