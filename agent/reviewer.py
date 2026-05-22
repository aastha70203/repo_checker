
"""LLM review client with schema repair, exponential backoff retries, and deterministic fallback."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List

from agent.models import CodeChunk, ReviewComment


ALLOWED_SEVERITIES = {"critical", "high", "medium", "low", "info"}
ALLOWED_CATEGORIES = {"bug", "security", "performance", "maintainability", "style", "test"}


SYSTEM_PROMPT = """You are a senior code review agent.
Return only valid JSON matching the user prompt's shape.

CRITICAL CONTEXT INSTRUCTIONS:
The code chunk you are reviewing has been semantically sliced from a larger file.
1. Decorators, imports, parent classes, module constants, or helper methods defined elsewhere are expected to be available at runtime. DO NOT flag them as "missing" or "undefined" unless they are obviously incorrect in their local context.
2. Focus strictly on logical bugs, security vulnerabilities, performance optimization, and formatting inside the provided boundaries.
3. Keep comments highly actionable and meaningful. Prefer no comment over a weak or false-positive comment."""


USER_PROMPT_TEMPLATE = """Review this code chunk.

File: {file_path}
Lines: {start_line}-{end_line}
Imports: {imports}
Symbols: {symbols}
Parse error: {parse_error}

Return JSON with this shape:
{{
  "comments": [
    {{
      "line": 12,
      "severity": "high|medium|low|info|critical",
      "category": "bug|security|performance|maintainability|style|test",
      "title": "short title",
      "comment": "why this matters",
      "suggestion": "specific fix",
      "confidence": 0
    }}
  ]
}}

Code:
```python
{code}
```"""


class CodeReviewer:
    """Review chunks using OpenAI or Google Gemini when configured, otherwise a local demo reviewer."""

    def __init__(self, model: str = "gpt-4o-mini", use_llm: bool = True, provider: str = "openai") -> None:
        # Realigned parameters keep original positional/keyword arguments backward compatible
        self.model = model
        self.use_llm = use_llm
        self.provider = provider.lower()

    def review_chunk(self, chunk: CodeChunk) -> List[ReviewComment]:
        if not self.use_llm:
            return self._heuristic_review(chunk)

        # Route dynamically based on chosen API provider
        if self.provider == "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                return self._retry_with_backoff(lambda: self._review_with_openai(chunk))
            except Exception as exc:
                return self._handle_fallback(chunk, f"OpenAI ({self.model}) rate limits or server issues. Error: {exc}")

        elif self.provider == "gemini" and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            try:
                return self._retry_with_backoff(lambda: self._review_with_gemini(chunk))
            except Exception as exc:
                return self._handle_fallback(chunk, f"Gemini API rate limits exceeded or key quota exhausted. Error: {exc}")

        return self._heuristic_review(chunk)

    def _retry_with_backoff(self, api_call_func: Any, retries: int = 5) -> Any:
        """Executes API calls with exponential backoff delays of 1s, 2s, 4s, 8s, 16s."""
        delays = [1, 2, 4, 8, 16]
        for attempt, delay in enumerate(delays):
            try:
                return api_call_func()
            except Exception as exc:
                # If we've run out of retries, raise the error to trigger fallback
                if attempt >= retries - 1 or attempt >= len(delays) - 1:
                    raise exc
                time.sleep(delay)

    def _handle_fallback(self, chunk: CodeChunk, error_msg: str) -> List[ReviewComment]:
        return [
            ReviewComment(
                file_path=chunk.file_path,
                line=chunk.start_line,
                severity="info",
                category="maintainability",
                title="LLM review unavailable",
                comment=f"The review agent encountered temporary connectivity issues: {error_msg}",
                suggestion="Check your API Key status or wait a moment for rate limits to reset.",
                confidence=45,
                source="fallback",
            )
        ] + self._heuristic_review(chunk)

    def _parse_json_defensively(self, text: str) -> Dict[str, Any]:
        """A robust defensive JSON extractor protecting against Markdown blocks or stray conversational tokens."""
        text = text.strip()
        if not text:
            return {}

        # 1. Attempt standard direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Extract contents matching outer brace parameters {...}
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # 3. Clean common Markdown artifacts and retry
        cleaned = re.sub(r"^```json|```python|```|^```", "", text, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # If all parsing attempts fail, return empty dict to avoid crashing the review process
        return {}



