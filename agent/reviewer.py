"""LLM review client with schema repair and deterministic fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from agent.models import CodeChunk, ReviewComment


ALLOWED_SEVERITIES = {"critical", "high", "medium", "low", "info"}
ALLOWED_CATEGORIES = {"bug", "security", "performance", "maintainability", "style", "test"}


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
    """Review chunks using OpenAI or Anthropic when configured, otherwise a local demo reviewer."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", use_llm: bool = True) -> None:
        self.provider = provider.lower()
        self.model = model
        self.use_llm = use_llm

    def review_chunk(self, chunk: CodeChunk) -> List[ReviewComment]:
        if not self.use_llm:
            return self._heuristic_review(chunk)

        # Route dynamically based on chosen API provider
        if self.provider == "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                return self._review_with_openai(chunk)
            except Exception as exc:
                return self._handle_fallback(chunk, f"OpenAI ({self.model}) failed: {exc}")

        elif self.provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            try:
                return self._review_with_anthropic(chunk)
            except Exception as exc:
                return self._handle_fallback(chunk, f"Anthropic ({self.model}) failed: {exc}")

        return self._heuristic_review(chunk)

    def _handle_fallback(self, chunk: CodeChunk, error_msg: str) -> List[ReviewComment]:
        return [
            ReviewComment(
                file_path=chunk.file_path,
                line=chunk.start_line,
                severity="info",
                category="maintainability",
                title="LLM review unavailable",
                comment=f"The raw model endpoint call raised an exception, falling back to local diagnostics. Error: {error_msg}",
                suggestion="Verify your Environment API configuration keys and rerun.",
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
        )
        content = response.choices[0].message.content or "{}"
        payload = json.loads(content)
        return self._comments_from_payload(payload, chunk, source="openai")

    def _review_with_anthropic(self, chunk: CodeChunk) -> List[ReviewComment]:
        import anthropic

        client = anthropic.Anthropic()
        # Claude expects structural JSON prompting enforcements via custom block suffixes or native systems
        user_prompt = self._build_user_prompt(chunk) + "\n\nIMPORTANT: Respond ONLY with the raw unadorned valid JSON object. Do not enclose it in markdown code blocks."
        
        response = client.messages.create(
            model=self.model,
            max_tokens=4000,
            temperature=0.1,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )
        
        content = response.content[0].text.strip()
        
        # Defensive JSON parsing mechanism if Claude wraps it in codeblock formats anyway
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?|\n?```$", "", content)
http://googleusercontent.com/immersive_entry_chip/0
3. In the sidebar, select **Anthropic**, pick **claude-3-5-sonnet-20241022**, and hit **Run Review** to watch Claude check your code chunks! Each suggestion card will state the source engine used (`⚙️ ANTHROPIC`).

Let me know when this is working nicely on your machine, and we will move to **Pizzazz #2: Before/After Git Diff patches**!