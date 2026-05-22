"""LLM review client with schema repair and deterministic fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from agent.models import CodeChunk, ReviewComment


ALLOWED_SEVERITIES = {"critical", "high", "medium", "low", "info"}
ALLOWED_CATEGORIES = {"bug", "security", "performance", "maintainability", "style", "test"}
GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


SYSTEM_PROMPT = """You are a senior code review agent.
Return only valid JSON matching the user prompt's shape. Review the supplied Python code chunk for actionable issues.
Do not invent context outside the chunk. Prefer no comment over a weak comment.
Every comment must include a self-rated confidence integer from 0 to 100.
Low confidence is acceptable when the issue needs human verification."""


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
    """Review chunks using Google AI Studio or OpenAI, with local fallback."""

    DEFAULT_MODELS = {
        "google": "gemini-2.5-flash",
        "openai": "gpt-4o-mini",
    }

    def __init__(self, provider: str = "google", model: str | None = None, use_llm: bool = True) -> None:
        self.provider = provider.lower()
        self.model = model or self.DEFAULT_MODELS.get(self.provider, self.DEFAULT_MODELS["google"])
        self.use_llm = use_llm

    def review_chunk(self, chunk: CodeChunk) -> List[ReviewComment]:
        if not self.use_llm:
            return self._heuristic_review(chunk)

        if self.provider == "google" and self._google_api_key():
            try:
                return self._review_with_google_ai_studio(chunk)
            except Exception as exc:
                return self._handle_fallback(chunk, f"Google AI Studio ({self.model}) failed: {exc}")

        if self.provider == "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                return self._review_with_openai(chunk)
            except Exception as exc:
                return self._handle_fallback(chunk, f"OpenAI ({self.model}) failed: {exc}")

        return self._heuristic_review(chunk)

    def _handle_fallback(self, chunk: CodeChunk, error_msg: str) -> List[ReviewComment]:
        return [
            ReviewComment(
                file_path=chunk.file_path,
                line=chunk.start_line,
                severity="info",
                category="maintainability",
                title="LLM review unavailable",
                comment=f"The model call failed, so this chunk used local diagnostics. Error: {error_msg}",
                suggestion="Verify API key configuration and rerun for model-backed comments.",
                confidence=45,
                source="fallback",
            )
        ] + self._heuristic_review(chunk)

    def _review_with_google_ai_studio(self, chunk: CodeChunk) -> List[ReviewComment]:
        import requests

        api_key = self._google_api_key()
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY for Google AI Studio.")

        response = requests.post(
            GOOGLE_API_URL.format(model=self.model),
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
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = self._parse_google_payload(response.json())
        return self._comments_from_payload(payload, chunk, source="google-ai-studio")

    def _parse_google_payload(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        candidates = response_json.get("candidates") or []
        if not candidates:
            return {}
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(str(part.get("text", "")) for part in parts).strip()
        return self._loads_json_object(text)

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
        )
        content = response.choices[0].message.content or "{}"
        payload = self._loads_json_object(content)
        return self._comments_from_payload(payload, chunk, source="openai")

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

    def _comments_from_payload(self, payload: Dict[str, Any], chunk: CodeChunk, source: str) -> List[ReviewComment]:
        raw_comments = payload.get("comments", [])
        if not isinstance(raw_comments, list):
            raw_comments = []
        comments: List[ReviewComment] = []
        for raw in raw_comments[:12]:
            if not isinstance(raw, dict):
                continue
            severity = str(raw.get("severity", "info")).lower()
            category = str(raw.get("category", "maintainability")).lower()
            comments.append(
                ReviewComment(
                    file_path=chunk.file_path,
                    line=self._clean_line(raw.get("line"), chunk),
                    severity=severity if severity in ALLOWED_SEVERITIES else "info",
                    category=category if category in ALLOWED_CATEGORIES else "maintainability",
                    title=str(raw.get("title") or "Review comment")[:120],
                    comment=str(raw.get("comment") or "").strip()[:1500],
                    suggestion=str(raw.get("suggestion") or "Review this area manually.").strip()[:1500],
                    confidence=self._clean_confidence(raw.get("confidence", 50)),
                    source=source,
                )
            )
        return [comment for comment in comments if comment.comment]

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

    def _google_api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def _loads_json_object(self, content: str) -> Dict[str, Any]:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
        try:
            loaded = json.loads(content or "{}")
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            loaded = json.loads(match.group(0)) if match else {}
        return loaded if isinstance(loaded, dict) else {}

    def _clean_confidence(self, value: Any) -> int:
        try:
            return max(0, min(100, int(float(value))))
        except (TypeError, ValueError):
            return 50

    def _clean_line(self, value: Any, chunk: CodeChunk) -> int:
        try:
            line = int(value)
        except (TypeError, ValueError):
            return chunk.start_line
        return max(chunk.start_line, min(chunk.end_line, line))
