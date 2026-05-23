from __future__ import annotations

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
    return [
        json.loads(line)
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def answer(question: str) -> dict[str, object]:
    rows = load_rows()
    q_tokens = tokens(question)
    scored = []
    for row in rows:
        instruction = str(row.get("instruction", ""))
        overlap = len(q_tokens & tokens(instruction))
        intent_bonus = (
            2 if str(row.get("intent", "")).replace("_", " ") in question.lower() else 0
        )
        scored.append((overlap + intent_bonus, row))
    score, row = max(scored, key=lambda item: item[0])
    if score < 2:
        categories = Counter(
            str(item.get("category", "")) for item in rows
        ).most_common(5)
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
