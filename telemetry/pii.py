from __future__ import annotations

import hashlib
import re
from typing import Any, Dict

# Basic detectors for high-risk PII patterns.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Phone-like numbers: allow country code, separators, and require at least 9 digits overall.
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
# Common personal ID shapes (e.g., SSN-like 3-2-4).
GOV_ID_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Keys that should never be logged verbatim.
SENSITIVE_FIELDS = {
    "messages",
    "transcript",
    "raw_transcript",
    "raw_messages",
    "full_conversation",
    "history",
    "lease_text",
    "lease_document",
    "raw_lease",
    "raw_lease_text",
    "full_lease",
    "pdf_base64",
    "raw_prompt",
    "raw_completion",
    "full_reply",
}


def _hash_token(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"[HASH:{digest}]"


def scrub_text(text: str) -> str:
    """Redact or hash obvious PII tokens from free text."""
    if not text:
        return text

    def _replace(match: re.Match, label: str) -> str:
        token = match.group(0)
        return f"[{label}_{_hash_token(token)}]"

    scrubbed = EMAIL_RE.sub(lambda m: _replace(m, "EMAIL"), text)
    scrubbed = PHONE_RE.sub(lambda m: _replace(m, "PHONE"), scrubbed)
    scrubbed = GOV_ID_RE.sub(lambda m: _replace(m, "ID"), scrubbed)
    return scrubbed


def _summarize_sequence(value: Any) -> Dict[str, Any]:
    length = len(value) if hasattr(value, "__len__") else None
    return {"redacted": True, "items": length}


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in SENSITIVE_FIELDS:
        return True
    if "lease" in lowered and ("text" in lowered or "content" in lowered or "body" in lowered):
        return True
    if "transcript" in lowered or "conversation" in lowered or "messages" in lowered:
        return True
    return False


def _scrub_collection(value: Any) -> Any:
    if isinstance(value, dict):
        return sanitize_log_payload(value)
    if isinstance(value, list):
        # If the list looks like chat messages, summarize instead of logging contents.
        if any(isinstance(item, dict) and "content" in item for item in value):
            return _summarize_sequence(value)
        return [scrub_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub_value(item) for item in value)
    return value


def scrub_value(value: Any) -> Any:
    """Scrub a generic value for PII before logging."""
    if isinstance(value, str):
        cleaned = scrub_text(value)
        if len(cleaned) > 500:
            return _hash_token(cleaned)
        return cleaned
    if isinstance(value, (dict, list, tuple)):
        return _scrub_collection(value)
    return value


def sanitize_log_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Strip or hash PII-heavy fields from a log payload."""
    if not isinstance(payload, dict):
        return {}

    cleaned: Dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            cleaned[key] = value
            continue
        if _is_sensitive_key(str(key)):
            cleaned[key] = _summarize_sequence(value)
            continue
        if isinstance(value, str) and len(value) > 500:
            cleaned[key] = _hash_token(scrub_text(value))
            continue
        cleaned[key] = scrub_value(value)
    return cleaned
