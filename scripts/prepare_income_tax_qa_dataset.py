"""Prepare and optionally upload the Income Tax Master QA dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from agent.tools.income_tax_qa import SOURCE_COLUMNS, build_income_tax_qa_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", required=True, help="Path to Income_Tax_Master.xlsx")
    parser.add_argument("--output-jsonl", required=True, help="Local JSONL output path")
    parser.add_argument("--sheet", default="Income Tax Master")
    parser.add_argument("--push-repo", help="Private HF dataset repo id to push")
    return parser.parse_args()


def load_rows(excel_path: str, sheet_name: str) -> list[dict]:
    frame = pd.read_excel(excel_path, sheet_name=sheet_name)
    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Workbook is missing required columns: {missing}")
    return frame.loc[:, SOURCE_COLUMNS].to_dict(orient="records")


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def push_private_dataset(repo_id: str, examples: list[dict]) -> None:
    from datasets import Dataset

    dataset = Dataset.from_list(examples)
    dataset.push_to_hub(repo_id, private=True)


def main() -> None:
    load_dotenv()
    args = parse_args()
    rows = load_rows(args.excel, args.sheet)
    examples = build_income_tax_qa_examples(rows)
    output_path = Path(args.output_jsonl)
    write_jsonl(output_path, examples)
    print(
        json.dumps(
            {
                "source_rows": len(rows),
                "qa_examples": len(examples),
                "output_jsonl": str(output_path),
                "push_repo": args.push_repo,
            },
            indent=2,
        )
    )
    if args.push_repo:
        push_private_dataset(args.push_repo, examples)


if __name__ == "__main__":
    main()
