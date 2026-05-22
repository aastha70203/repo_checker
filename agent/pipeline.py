"""End-to-end orchestration for the AI code review agent."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from agent.cloner import RepoCloner
from agent.models import ReviewComment, ReviewRun
from agent.parser import PythonASTParser
from agent.reviewer import CodeReviewer


class CodeReviewAgent:
    """Coordinates ingestion, AST parsing, review, and structured output."""

    def __init__(
        self,
        base_dir: str | None = None,
        max_chunks: int = 50,
        use_llm: bool = True,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.cloner = RepoCloner(base_dir=base_dir)
        self.parser = PythonASTParser()
        self.reviewer = CodeReviewer(model=model, use_llm=use_llm)
        self.max_chunks = max_chunks

    def review_repository(
        self,
        repo_url: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> ReviewRun:
        def notify(message: str) -> None:
            if progress_callback:
                progress_callback(message)

        clone_result = self.cloner.clone(repo_url, progress_callback=notify)
        if not clone_result.success:
            return ReviewRun(
                repo_url=repo_url,
                repo_path=Path(),
                repo_name=clone_result.repo_name,
                branch="",
                chunks_reviewed=0,
                comments=[],
                errors=[clone_result.error or "Clone failed."],
            )

        notify("Parsing Python files with ast...")
        chunks = self.parser.parse_repository(clone_result.repo_path)
        warnings = list(clone_result.warnings)
        if not chunks:
            warnings.append("No reviewable Python chunks were found.")

        if len(chunks) > self.max_chunks:
            warnings.append(f"Found {len(chunks)} chunks; reviewed the first {self.max_chunks}.")
            chunks = chunks[: self.max_chunks]

        comments: List[ReviewComment] = []
        for index, chunk in enumerate(chunks, start=1):
            notify(f"Reviewing chunk {index}/{len(chunks)}: {chunk.file_path}:{chunk.start_line}")
            comments.extend(self.reviewer.review_chunk(chunk))

        notify(f"Review complete: {len(comments)} comments.")
        return ReviewRun(
            repo_url=repo_url,
            repo_path=clone_result.repo_path,
            repo_name=clone_result.repo_name,
            branch=clone_result.branch,
            chunks_reviewed=len(chunks),
            comments=comments,
            warnings=warnings,
        )

    def cleanup(self) -> None:
        self.cloner.cleanup()
