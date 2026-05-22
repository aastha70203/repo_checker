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

    DEFAULT_MODELS = {
        "google": "gemini-2.5-flash",
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4o-mini",
    }

    def __init__(
        self,
        provider: str = "google",
        model: str | None = None,
        use_llm: bool = True,
        max_retries: int = 2,
    ) -> None:
        # Normalize and store provider
        self.provider = provider.lower() if provider else "google"
        self.use_llm = use_llm
        self.max_retries = max_retries
        self.model = model or self.DEFAULT_MODELS.get(self.provider, "gemini-2.5-flash")

    def review_chunk(self, chunk: CodeChunk) -> List[ReviewComment]:
        if not self.use_llm:
            return self._heuristic_review(chunk)

        # Route dynamically based on chosen API provider
        if self.provider == "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                return self._retry_with_backoff(lambda: self._review_with_openai(chunk))
            except Exception as exc:
                return self._handle_fallback(chunk, f"OpenAI ({self.model}) rate limits or server issues. Error: {exc}")

        elif self.provider in {"google", "gemini"} and self._google_api_key():
            try:
                return self._retry_with_backoff(lambda: self._review_with_gemini(chunk))
            except Exception as exc:
                return self._handle_fallback(chunk, f"Gemini API rate limits exceeded or key quota exhausted. Error: {exc}")

        return self._heuristic_review(chunk)

    def provider_configured(self) -> bool:
        if self.provider in {"google", "gemini"}:
            return bool(self._google_api_key())
        if self.provider == "openai":
            return bool(os.getenv("OPENAI_API_KEY"))
        return False

    def _retry_with_backoff(self, api_call_func: Any) -> Any:
        """Executes API calls with exponential backoff delays of 1s, 2s, 4s, 8s, 16s."""
        delays = [1, 2, 4, 8, 16]
        # Restrict attempts based on max_retries or delays length
        retries = min(self.max_retries + 1, len(delays))
        for attempt in range(retries):
            try:
                return api_call_func()
            except Exception as exc:
                if attempt >= retries - 1:
                    raise exc
                time.sleep(delays[attempt])

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

    def _review_with_openai(self, chunk: CodeChunk) -> List[ReviewComment]:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(chunk)},
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
        )
        content = response.choices[0].message.content or "{}"
        payload = self._parse_json_defensively(content)
        return self._comments_from_payload(payload, chunk, source="openai")

    def _review_with_gemini(self, chunk: CodeChunk) -> List[ReviewComment]:
        import requests

        api_key = self._google_api_key()
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY for Google AI Studio.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            json={
                "systemInstruction": {
                    "parts": [{"text": SYSTEM_PROMPT}],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": self._build_user_prompt(chunk)}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "responseMimeType": "application/json",
                    "maxOutputTokens": 1200,
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = self._parse_google_payload(response.json())
        return self._comments_from_payload(payload, chunk, source="google-ai-studio")

    def _google_api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def _parse_google_payload(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        candidates = response_json.get("candidates") or []
        if not candidates:
            return {}
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(str(part.get("text", "")) for part in parts).strip()
        return self._parse_json_defensively(text)

    def _comments_from_payload(self, payload: Dict[str, Any], chunk: CodeChunk, source: str) -> List[ReviewComment]:
        from agent.schema import validate_review_payload
        return validate_review_payload(payload, chunk, source)

    def _build_user_prompt(self, chunk: CodeChunk) -> str:
        symbols = [f"{s.kind}:{s.name}@{s.start_line}-{s.end_line}" for s in chunk.symbols]
        return USER_PROMPT_TEMPLATE.format(
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            imports=", ".join(chunk.imports) or "none",
            symbols=", ".join(symbols) or "none",
            parse_error=chunk.parse_error or "none",
            code=chunk.code[:16_000],
        )

    def _heuristic_review(self, chunk: CodeChunk) -> List[ReviewComment]:
        comments: List[ReviewComment] = []
        if chunk.parse_error:
            comments.append(
                ReviewComment(
                    file_path=chunk.file_path,
                    line=chunk.start_line,
                    severity="high",
                    category="bug",
                    title="Python syntax error",
                    comment="This file could not be parsed with Python ast, so runtime/import behavior is likely broken.",
                    suggestion=f"Fix the syntax issue reported by the parser: {chunk.parse_error}",
                    confidence=92,
                    source="heuristic",
                )
            )

        patterns = [
            (r"\beval\s*\(", "critical", "security", "Use of eval", "eval can execute arbitrary code when input is not fully trusted.", "Replace eval with a safe parser or explicit dispatch table.", 88),
            (r"\bexec\s*\(", "critical", "security", "Use of exec", "exec can execute arbitrary code and is difficult to audit.", "Avoid exec or strictly constrain the executed input.", 86),
            (r"subprocess\.[\w_]+\([^\n]*shell\s*=\s*True", "high", "security", "Shell execution enabled", "shell=True can allow command injection when arguments include user-controlled data.", "Pass arguments as a list and keep shell=False.", 82),
            (r"except\s+Exception\s*:\s*(?:\n\s*)?pass\b", "medium", "maintainability", "Swallowed exception", "A broad exception handler that silently passes can hide real failures.", "Log the exception or catch a narrower error.", 78),
            (r"TODO|FIXME", "info", "maintainability", "Open TODO marker", "The chunk contains an unresolved TODO/FIXME marker.", "Track or resolve the marker before release.", 55),
        ]
        for pattern, severity, category, title, comment, suggestion, confidence in patterns:
            for match in re.finditer(pattern, chunk.code, flags=re.IGNORECASE | re.MULTILINE):
                line = chunk.start_line + chunk.code[: match.start()].count("\n")
                comments.append(
                    ReviewComment(
                        file_path=chunk.file_path,
                        line=line,
                        severity=severity,
                        category=category,
                        title=title,
                        comment=comment,
                        suggestion=suggestion,
                        confidence=confidence,
                        source="heuristic",
                    )
                )
        return comments[:12]

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
