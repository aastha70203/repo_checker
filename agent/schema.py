"""Schema validation and normalization for LLM review payloads."""

from __future__ import annotations

from typing import Any, Dict, List

from agent.models import CodeChunk, ReviewComment


ALLOWED_SEVERITIES = {"critical", "high", "medium", "low", "info"}
ALLOWED_CATEGORIES = {"bug", "security", "performance", "maintainability", "style", "test"}
MAX_COMMENTS_PER_CHUNK = 12


def validate_review_payload(payload: Dict[str, Any], chunk: CodeChunk, source: str) -> List[ReviewComment]:
    """Convert model JSON into safe ReviewComment objects.

    The LLM is asked for a strict JSON shape, but production code should still
    treat model output as untrusted input. This function accepts only comment
    dictionaries with useful text, normalizes enum-like fields, clamps line and
    confidence values, and drops malformed entries.
    """
    raw_comments = payload.get("comments", [])
    if not isinstance(raw_comments, list):
        return []

    comments: List[ReviewComment] = []
    for raw in raw_comments[:MAX_COMMENTS_PER_CHUNK]:
        if not isinstance(raw, dict):
            continue

        body = _clean_text(raw.get("comment"), max_length=1500)
        if not body:
            continue

        severity = _clean_enum(raw.get("severity"), ALLOWED_SEVERITIES, default="info")
        category = _clean_enum(raw.get("category"), ALLOWED_CATEGORIES, default="maintainability")
        comments.append(
            ReviewComment(
                file_path=chunk.file_path,
                line=_clean_line(raw.get("line"), chunk),
                severity=severity,
                category=category,
                title=_clean_text(raw.get("title"), default="Review comment", max_length=120),
                comment=body,
                suggestion=_clean_text(raw.get("suggestion"), default="Review this area manually.", max_length=1500),
                confidence=_clean_confidence(raw.get("confidence", 50)),
                source=source,
            )
        )

    return comments


def _clean_enum(value: Any, allowed: set[str], default: str) -> str:
    cleaned = str(value or default).strip().lower()
    return cleaned if cleaned in allowed else default


def _clean_text(value: Any, default: str = "", max_length: int = 1500) -> str:
    text = str(value if value is not None else default).strip()
    return text[:max_length]


def _clean_confidence(value: Any) -> int:
    try:
        val = float(value)
        if 0.0 < val <= 1.0:
            val *= 100
        return max(0, min(100, int(val)))
    except (TypeError, ValueError):
        return 50


def _clean_line(value: Any, chunk: CodeChunk) -> int:
    try:
        line = int(value)
    except (TypeError, ValueError):
        return chunk.start_line
    return max(chunk.start_line, min(chunk.end_line, line))
