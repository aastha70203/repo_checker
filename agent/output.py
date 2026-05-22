"""Report generation and optional GitHub PR comment posting."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, Iterable, List

from agent.models import ReviewComment, ReviewRun


def comments_to_markdown(comments: Iterable[ReviewComment], title: str = "AI Code Review") -> str:
    comments = list(comments)
    lines = [f"# {title}", ""]
    if not comments:
        lines.append("No review comments were generated.")
        return "\n".join(lines)

    severity_counts = Counter(c.severity for c in comments)
    lines.append("## Summary")
    for severity, count in sorted(severity_counts.items()):
        lines.append(f"- {severity}: {count}")
    lines.append("")

    for comment in sorted(comments, key=lambda c: (c.file_path, c.line, c.severity)):
        verify = " [verify this]" if comment.verify_label else ""
        lines.extend(
            [
                f"## {comment.file_path}:{comment.line} - {comment.title}{verify}",
                f"- Severity: {comment.severity}",
                f"- Category: {comment.category}",
                f"- Confidence: {comment.confidence}%",
                f"- Source: {comment.source}",
                "",
                comment.comment,
                "",
                f"Suggestion: {comment.suggestion}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def run_to_json(run: ReviewRun) -> str:
    return json.dumps(run.to_dict(), indent=2)


def post_pr_comments(
    repo_full_name: str,
    pull_number: int,
    commit_sha: str,
    comments: List[ReviewComment],
    token: str | None = None,
) -> Dict[str, int]:
    """Post review comments to a GitHub pull request using the REST API.

    This is intentionally optional. It runs only when the user supplies a token,
    repo, PR number, and commit SHA.
    """
    token = token or os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN is required to post PR comments.")

    import requests

    created = 0
    skipped = 0
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pull_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    for comment in comments:
        if comment.confidence < 60:
            skipped += 1
            continue
        body = (
            f"**{comment.title}**\n\n"
            f"{comment.comment}\n\n"
            f"Suggestion: {comment.suggestion}\n\n"
            f"Severity: `{comment.severity}` | Confidence: `{comment.confidence}%`"
        )
        response = requests.post(
            url,
            headers=headers,
            json={
                "body": body,
                "commit_id": commit_sha,
                "path": comment.file_path,
                "line": comment.line,
                "side": "RIGHT",
            },
            timeout=20,
        )
        if response.status_code in {200, 201}:
            created += 1
        else:
            skipped += 1
    return {"created": created, "skipped": skipped}
