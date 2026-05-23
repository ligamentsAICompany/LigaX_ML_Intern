"""Deterministic post-training quality gate over golden eval cases."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal

from agent.core.golden_eval import NOT_SPECIFIED
from agent.core.redact import scrub, scrub_string

EvalStatus = Literal["passed", "failed", "needs_rag", "needs_more_data"]

_WORD_RE = re.compile(r"[a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_MISSING_CONTEXT_MARKERS = (
    "need retrieval",
    "need rag",
    "need context",
    "need source",
    "needs retrieval",
    "needs context",
    "needs source",
    "not enough context",
    "without context",
    "cannot answer",
    "can't answer",
    "do not know",
    "don't know",
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "used",
    "with",
}
_GENERIC_GROUNDED_WORDS = {
    "according",
    "answer",
    "based",
    "context",
    "given",
    "information",
    "provided",
    "provides",
    "source",
    "states",
    "table",
}
_INCOME_TAX_UNSUPPORTED_TOPICS = {
    "acid": "chemistry",
    "acids": "chemistry",
    "base": "chemistry",
    "bases": "chemistry",
    "catalyst": "chemistry",
    "catalysts": "chemistry",
    "chemistry": "chemistry",
    "laboratory": "chemistry",
    "reaction": "chemistry",
    "reactions": "chemistry",
    "irs": "us_tax",
    "w2": "us_tax",
    "wage": "us_tax",
    "wages": "us_tax",
    "employer": "us_tax",
    "employers": "us_tax",
}


def evaluate_post_training_outputs(
    *,
    cases: list[Mapping[str, Any]],
    outputs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Score model output strings against source-grounded golden eval cases.

    This gate is intentionally conservative and does not execute a live model.
    Callers provide captured outputs keyed by golden eval case id.
    """

    normalized_cases = [dict(case) for case in cases if isinstance(case, Mapping)]
    normalized_outputs = dict(outputs or {})
    results = [
        _evaluate_case(case, output=normalized_outputs.get(str(case.get("id", ""))))
        for case in normalized_cases
    ]
    summary = Counter(result["status"] for result in results)
    for status in ("passed", "failed", "needs_rag", "needs_more_data"):
        summary.setdefault(status, 0)

    report = {
        "status": _overall_status(results),
        "case_count": len(results),
        "summary": dict(summary),
        "cases": results,
    }
    return scrub(report)


def summarize_post_training_eval(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a compact scrubbed summary suitable for provenance payloads."""

    if not isinstance(report, Mapping):
        return {}
    summary = (
        report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    )
    cases = report.get("cases") if isinstance(report.get("cases"), list) else []
    reasons = []
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        for reason in case.get("reasons") or []:
            text = str(reason)
            if text and text not in reasons:
                reasons.append(text)
            if len(reasons) >= 3:
                break
        if len(reasons) >= 3:
            break
    valid = _valid_eval_report(report, cases=cases, summary=summary)
    status = report.get("status") if valid else "unverified"
    return scrub(
        {
            "status": status,
            "valid": valid,
            "case_count": report.get("case_count", len(cases)),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "needs_rag": summary.get("needs_rag", 0),
            "needs_more_data": summary.get("needs_more_data", 0),
            "top_reasons": reasons,
        }
    )


def _evaluate_case(case: Mapping[str, Any], *, output: Any) -> dict[str, Any]:
    case_id = str(case.get("id") or "")
    question = scrub_string(str(case.get("question") or ""))
    expected_answer = scrub_string(str(case.get("expected_answer") or ""))
    safe_output = scrub(output)
    output_text = scrub_string(str(safe_output).strip()) if output is not None else ""
    expected_facts = _expected_facts(case)
    normalized_output = _normalize(output_text)

    base = {
        "case_id": case_id,
        "question": question,
        "expected_answer": expected_answer,
        "model_output": _truncate(output_text),
        "missing_facts": [],
        "reasons": [],
    }
    if not output_text:
        return {
            **base,
            "status": "needs_more_data",
            "score": 0.0,
            "reasons": ["No model output was provided for this golden eval case."],
        }

    missing_facts = [
        fact for fact in expected_facts if not _fact_present(fact, normalized_output)
    ]

    if missing_facts and _asks_for_retrieval(normalized_output):
        return {
            **base,
            "status": "needs_rag",
            "score": 0.0,
            "missing_facts": expected_facts,
            "reasons": [
                "Model output asks for retrieval/source context instead of answering."
            ],
        }

    unsupported_topics = _unsupported_topics(case=case, output=output_text)
    unsupported_claims = _unsupported_claims(case=case, output=output_text)
    if missing_facts or unsupported_topics or unsupported_claims:
        reasons = []
        if missing_facts:
            reasons.append(
                "Missing expected facts: "
                + ", ".join(scrub_string(f) for f in missing_facts)
            )
        if unsupported_topics:
            reasons.append(
                "Off-topic or unsupported topics: " + ", ".join(unsupported_topics)
            )
        if unsupported_claims:
            reasons.append("Unsupported claims: " + ", ".join(unsupported_claims))
        score = _score(expected_facts=expected_facts, missing_facts=missing_facts)
        return {
            **base,
            "status": "failed",
            "score": score,
            "missing_facts": missing_facts,
            "unsupported_topics": unsupported_topics,
            "unsupported_claims": unsupported_claims,
            "reasons": reasons,
        }

    return {
        **base,
        "status": "passed",
        "score": 1.0,
        "reasons": ["All expected source-grounded facts were present."],
    }


def _overall_status(results: list[Mapping[str, Any]]) -> EvalStatus:
    if not results:
        return "needs_more_data"
    statuses = {str(result.get("status")) for result in results}
    if "failed" in statuses:
        return "failed"
    if statuses == {"passed"}:
        return "passed"
    if "needs_more_data" in statuses:
        return "needs_more_data"
    if "needs_rag" in statuses:
        return "needs_rag"
    return "failed"


def _expected_facts(case: Mapping[str, Any]) -> list[str]:
    expected_answer = str(case.get("expected_answer") or "").strip()
    facts: list[str] = []
    if expected_answer:
        facts.append(expected_answer)

    source_metadata = (
        case.get("source_metadata")
        if isinstance(case.get("source_metadata"), Mapping)
        else {}
    )
    source_fields = (
        source_metadata.get("source_fields")
        if isinstance(source_metadata.get("source_fields"), Mapping)
        else {}
    )
    for value in source_fields.values():
        if (
            isinstance(value, str)
            and value.strip()
            and value.strip() == expected_answer
        ):
            facts.append(value.strip())

    deduped: list[str] = []
    for fact in facts:
        text = scrub_string(fact.strip())
        if text and text not in deduped:
            deduped.append(text)
    return deduped or [NOT_SPECIFIED]


def _unsupported_topics(*, case: Mapping[str, Any], output: str) -> list[str]:
    domain = str(case.get("domain") or "").lower()
    if domain != "income_tax":
        return []
    expected_and_source = _normalize(
        " ".join(
            [
                str(case.get("question") or ""),
                str(case.get("expected_answer") or ""),
                str(case.get("source_metadata") or ""),
            ]
        )
    )
    output_tokens = set(_WORD_RE.findall(_normalize(output)))
    topics = {
        label
        for token, label in _INCOME_TAX_UNSUPPORTED_TOPICS.items()
        if token in output_tokens
        and token not in set(_WORD_RE.findall(expected_and_source))
    }
    if "w" in output_tokens and "2" in output_tokens:
        topics.add("w-2")
    return sorted(topics)


def _unsupported_claims(*, case: Mapping[str, Any], output: str) -> list[str]:
    supported = _content_tokens(
        " ".join(
            [
                str(case.get("question") or ""),
                str(case.get("expected_answer") or ""),
                str(case.get("source_metadata") or ""),
            ]
        )
    )
    claims: list[str] = []
    for sentence in _SENTENCE_RE.split(output):
        unsupported = sorted(_content_tokens(sentence) - supported)
        if len(unsupported) >= 2:
            claims.append(" ".join(unsupported[:4]))
    return claims


def _asks_for_retrieval(normalized_output: str) -> bool:
    return any(marker in normalized_output for marker in _MISSING_CONTEXT_MARKERS)


def _valid_eval_report(
    report: Mapping[str, Any],
    *,
    cases: list[Any],
    summary: Mapping[str, Any],
) -> bool:
    status = report.get("status")
    if status not in {"passed", "failed", "needs_rag", "needs_more_data"}:
        return False
    if not isinstance(report.get("case_count"), int):
        return False
    if report.get("case_count") != len(cases):
        return False

    case_statuses = [case.get("status") for case in cases if isinstance(case, Mapping)]
    if len(case_statuses) != len(cases):
        return False
    expected_summary = Counter(str(case_status) for case_status in case_statuses)
    for eval_status in ("passed", "failed", "needs_rag", "needs_more_data"):
        if summary.get(eval_status, 0) != expected_summary.get(eval_status, 0):
            return False
    return status == _overall_status(cases)


def _fact_present(fact: str, normalized_output: str) -> bool:
    normalized_fact = _normalize(fact)
    if normalized_fact in normalized_output:
        return True

    fact_tokens = _content_tokens(fact)
    if not 2 <= len(fact_tokens) <= 6:
        return False
    output_tokens = _content_tokens(normalized_output)
    overlap = len(fact_tokens & output_tokens) / len(fact_tokens)
    return overlap >= 0.8


def _content_tokens(text: str) -> set[str]:
    return {
        _stem_token(token)
        for token in _WORD_RE.findall(_normalize(text))
        if token not in _STOPWORDS and token not in _GENERIC_GROUNDED_WORDS
    }


def _stem_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _score(*, expected_facts: list[str], missing_facts: list[str]) -> float:
    if not expected_facts:
        return 0.0
    present = len(expected_facts) - len(missing_facts)
    return round(max(0.0, present / len(expected_facts)), 3)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    collapsed = _WHITESPACE_RE.sub(" ", ascii_text.replace("-", " ")).strip()
    return collapsed


def _truncate(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
