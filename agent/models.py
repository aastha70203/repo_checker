"""Shared data models for the review pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


LOW_CONFIDENCE_THRESHOLD = 60


@dataclass
class CodeSymbol:
    name: str
    kind: str
    start_line: int
    end_line: int


@dataclass
class CodeChunk:
    file_path: str
    language: str
    start_line: int
    end_line: int
    code: str
    symbols: List[CodeSymbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    parse_error: Optional[str] = None

    @property
    def line_count(self) -> int:
        return max(0, self.end_line - self.start_line + 1)


@dataclass
class ReviewComment:
    file_path: str
    line: int
    severity: str
    category: str
    title: str
    comment: str
    suggestion: str
    confidence: int
    source: str = "llm"

    @property
    def verify_label(self) -> str:
        return "verify this" if self.confidence < LOW_CONFIDENCE_THRESHOLD else ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["verify_label"] = self.verify_label
        return data


@dataclass
class ReviewRun:
    repo_url: str
    repo_path: Path
    repo_name: str
    branch: str
    chunks_reviewed: int
    comments: List[ReviewComment]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def high_confidence_comments(self) -> List[ReviewComment]:
        return [c for c in self.comments if c.confidence >= LOW_CONFIDENCE_THRESHOLD]

    @property
    def low_confidence_comments(self) -> List[ReviewComment]:
        return [c for c in self.comments if c.confidence < LOW_CONFIDENCE_THRESHOLD]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_url": self.repo_url,
            "repo_path": str(self.repo_path),
            "repo_name": self.repo_name,
            "branch": self.branch,
            "chunks_reviewed": self.chunks_reviewed,
            "comments": [c.to_dict() for c in self.comments],
            "warnings": self.warnings,
            "errors": self.errors,
        }
