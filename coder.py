"""Convenience CLI for running the code review agent locally."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.output import comments_to_markdown, run_to_json
from agent.pipeline import CodeReviewAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AI code review agent.")
    parser.add_argument("repo", help="GitHub repo URL or owner/repo shorthand")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic local checks only")
    parser.add_argument("--max-chunks", type=int, default=50)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown")
    parser.add_argument("--output", type=Path, help="Optional output file")
    args = parser.parse_args()

    agent = CodeReviewAgent(max_chunks=args.max_chunks, use_llm=not args.no_llm)
    run = agent.review_repository(args.repo, progress_callback=lambda msg: print(f"[agent] {msg}"))
    rendered = run_to_json(run) if args.json else comments_to_markdown(run.comments, f"AI Code Review - {run.repo_name}")
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(rendered)
    return 0 if not run.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
