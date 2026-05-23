"""Central redaction utilities for telemetry and saved traces.

The scrubber is intentionally best-effort and conservative: it targets common
credential formats plus obvious Indian PII identifiers without trying to infer
free-form personal data from arbitrary prose.
"""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "[REDACTED_HF_TOKEN]"),
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "[REDACTED_ANTHROPIC_KEY]"),
    (re.compile(r"sk-(?!ant-)[A-Za-z0-9_\-]{40,}"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{36,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "[REDACTED_AWS_KEY_ID]"),
    (
        re.compile(r"(?i)\baws_secret_access_key\s*[:=]\s*([A-Za-z0-9/+=]{30,})"),
        "AWS_SECRET_ACCESS_KEY=[REDACTED_SECRET]",
    ),
    (
        re.compile(r"(?i)\b(accountkey|azure_client_secret)\s*[:=]\s*[^;\s]+"),
        r"\1=[REDACTED_SECRET]",
    ),
    (
        re.compile(r"-----BEGIN PRIVATE KEY-----.*?-----END PRIVATE KEY-----", re.S),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.=]{20,}"), "Bearer [REDACTED]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[REDACTED_AADHAAR]"),
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "[REDACTED_PAN]"),
    (
        re.compile(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"),
        "[REDACTED_GSTIN]",
    ),
    (re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), "[REDACTED_IFSC]"),
    (
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
        "[REDACTED_PHONE]",
    ),
    (
        re.compile(r"(?i)\b(?:invoice|inv)[\s:#-]*[A-Z0-9][A-Z0-9_\-/]{3,}\b"),
        "[REDACTED_INVOICE]",
    ),
    (
        re.compile(
            r"(?i)\b(customer|client)\s+(?:name\s*[:=]?\s*)?(?!at\b)[A-Z][A-Za-z.'-]*(?:\s+(?!at\b)[A-Z][A-Za-z.'-]*){0,2}"
        ),
        r"\1 [REDACTED_NAME]",
    ),
    (
        re.compile(
            r"(?i)\b(address|addr|shipping address)\s*[:=]\s*[A-Za-z0-9][A-Za-z0-9 .,#/\-]{5,80}"
        ),
        r"\1=[REDACTED_ADDRESS]",
    ),
    (
        re.compile(r"(?i)\bat\s+\d{1,5}\s+[A-Za-z0-9 .,#/\-]{3,40}\b"),
        "at [REDACTED_ADDRESS]",
    ),
    (
        re.compile(
            r"(?i)\b(?:bank\s*)?(?:account|acct)\s*(?:no\.?|number)?\s*[:#=]?\s*\d{9,18}\b"
        ),
        "[REDACTED_BANK_ACCOUNT]",
    ),
]

_SECRETY_NAMES = re.compile(
    r"(?i)\b("
    r"HF_TOKEN|HUGGINGFACE(?:HUB|_HUB)?_TOKEN|HUGGINGFACEHUB_API_TOKEN|"
    r"ANTHROPIC_API_KEY|OPENAI_API_KEY|GITHUB_TOKEN|GH_TOKEN|"
    r"AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|AZURE_CLIENT_SECRET|"
    r"GOOGLE_APPLICATION_CREDENTIALS|GCP_SERVICE_ACCOUNT_KEY|"
    r"PASSWORD|PASSWD|SECRET|API_KEY|ACCESS_TOKEN|REFRESH_TOKEN|PRIVATE_KEY"
    r")\s*[:=]\s*([^\s\"']+)"
)

_MAX_SCRUB_DEPTH = 50
_CYCLE_MARKER = "[REDACTED_CYCLE]"
_MAX_DEPTH_MARKER = "[REDACTED_MAX_DEPTH]"
_SAFE_TOKEN_METRIC_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
}
_SECRET_KEY_NAMES = re.compile(
    r"(?i)(^|[_\-\s])("
    r"token|hf[_\-\s]?token|authorization(?:[_\-\s]?token)?|"
    r"secret|password|passwd|api[_\-\s]?key|access[_\-\s]?key(?:[_\-\s]?id)?|"
    r"client[_\-\s]?id|tenant[_\-\s]?id|subscription[_\-\s]?id|"
    r"access[_\-\s]?token|refresh[_\-\s]?token|bearer|private[_\-\s]?key|credentials?"
    r")($|[_\-\s])"
)


def _env_replacement(match: re.Match[str]) -> str:
    name = match.group(1)
    value = match.group(2)
    redacted_value = scrub_string(value)
    if redacted_value != value:
        return f"{name}={redacted_value}"
    if value.startswith("[REDACTED_"):
        return f"{name}={value}"
    return f"{name}=[REDACTED_SECRET]"


def scrub_string(s: str) -> str:
    """Apply all redaction patterns to a single string."""
    if not isinstance(s, str) or not s:
        return s
    out = s
    for pattern, replacement in _PATTERNS:
        out = pattern.sub(replacement, out)
    return _SECRETY_NAMES.sub(_env_replacement, out)


def _is_secret_key(key: str, value: Any) -> bool:
    if key.lower() in _SAFE_TOKEN_METRIC_KEYS:
        return False
    if (
        "token" in key.lower()
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return False
    return bool(_SECRET_KEY_NAMES.search(key))


def scrub(obj: Any, *, max_depth: int = _MAX_SCRUB_DEPTH) -> Any:
    """Recursively scrub strings in a JSON-like payload without mutating input."""
    return _scrub(obj, max_depth=max_depth, depth=0, seen=set())


def _scrub(obj: Any, *, max_depth: int, depth: int, seen: set[int]) -> Any:
    if isinstance(obj, str):
        return scrub_string(obj)
    if depth >= max_depth:
        return _MAX_DEPTH_MARKER
    if isinstance(obj, dict):
        obj_id = id(obj)
        if obj_id in seen:
            return _CYCLE_MARKER
        seen.add(obj_id)
        scrubbed: dict[Any, Any] = {}
        try:
            for key, value in obj.items():
                if isinstance(key, str) and _is_secret_key(key, value):
                    scrubbed[key] = "[REDACTED_SECRET]"
                else:
                    scrubbed[key] = _scrub(
                        value, max_depth=max_depth, depth=depth + 1, seen=seen
                    )
            return scrubbed
        finally:
            seen.remove(obj_id)
    if isinstance(obj, list):
        obj_id = id(obj)
        if obj_id in seen:
            return _CYCLE_MARKER
        seen.add(obj_id)
        try:
            return [
                _scrub(value, max_depth=max_depth, depth=depth + 1, seen=seen)
                for value in obj
            ]
        finally:
            seen.remove(obj_id)
    if isinstance(obj, tuple):
        obj_id = id(obj)
        if obj_id in seen:
            return _CYCLE_MARKER
        seen.add(obj_id)
        try:
            return tuple(
                _scrub(value, max_depth=max_depth, depth=depth + 1, seen=seen)
                for value in obj
            )
        finally:
            seen.remove(obj_id)
    if isinstance(obj, set):
        obj_id = id(obj)
        if obj_id in seen:
            return _CYCLE_MARKER
        seen.add(obj_id)
        try:
            return {
                _scrub(value, max_depth=max_depth, depth=depth + 1, seen=seen)
                for value in obj
            }
        finally:
            seen.remove(obj_id)
    return obj
