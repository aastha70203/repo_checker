"""Report generation and optional GitHub PR comment posting."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List

from agent.models import ReviewComment, ReviewRun

# Logical priority sorting for severity levels
SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4
}


def comments_to_markdown(comments: Iterable[ReviewComment], title: str = "AI Code Review") -> str:
    """Formats findings into a consistent, severity-sorted Markdown report."""
    comments = list(comments)
    lines = [f"# {title}", ""]
    if not comments:
        lines.append("No review comments were generated.")
        return "\n".join(lines)

    # Compute severity frequencies
    severity_counts: Dict[str, int] = {}
    for c in comments:
        sev = c.severity.lower()
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    lines.append("## Summary")
    # Sort severities logically by severity order
    sorted_sevs = sorted(
        severity_counts.items(),
        key=lambda item: SEVERITY_ORDER.get(item[0], 99)
    )
    for severity, count in sorted_sevs:
        lines.append(f"- {severity}: {count}")
    lines.append("")

    # Sort comments logically by severity rank, file path, and then line number
    sorted_comments = sorted(
        comments,
        key=lambda c: (SEVERITY_ORDER.get(c.severity.lower(), 99), c.file_path, c.line)
    )

    for comment in sorted_comments:
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
    """Defensively serializes the pipeline run details into raw JSON format."""
    try:
        return json.dumps(run.to_dict(), indent=2)
    except Exception as exc:
        return json.dumps({
            "error": f"JSON serialization failed: {exc}",
            "repo_url": run.repo_url,
            "repo_name": run.repo_name,
        }, indent=2)


def post_pr_comments(
    repo_full_name: str,
    pull_number: int,
    comments: List[ReviewComment],
    commit_sha: str | None = None,
    token: str | None = None,
) -> Dict[str, Any]:
    """Post review comments to a GitHub pull request using the REST API.

    The commit_sha is now optional. If not provided, it is automatically fetched
    from the GitHub Pull Request API using the supplied token.
    """
    token = token or os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN is required to post PR comments.")

    import requests

    created = 0
    skipped = 0
    failed_details: List[str] = []
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pull_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Automatically fetch Commit SHA if not supplied
    if not commit_sha:
        pr_url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pull_number}"
        try:
            pr_response = requests.get(pr_url, headers=headers, timeout=15)
            if pr_response.status_code == 200:
                pr_data = pr_response.json()
                commit_sha = pr_data.get("head", {}).get("sha")
            else:
                raise ValueError(
                    f"Failed to fetch PR head commit from GitHub API (HTTP {pr_response.status_code}): {pr_response.text}"
                )
        except Exception as exc:
            raise ValueError(f"Could not automatically resolve Commit SHA: {exc}")

    if not commit_sha:
        raise ValueError("Commit SHA could not be determined. Please supply it manually.")

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
        try:
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
                try:
                    err_msg = response.json().get("message", response.text)
                except Exception:
                    err_msg = response.text

                if response.status_code == 422:
                    failed_details.append(
                        f"Could not post '{comment.title}' to {comment.file_path}:{comment.line}. "
                        "Reason: Line/file is not part of the Pull Request diff."
                    )
                else:
                    failed_details.append(
                        f"Could not post '{comment.title}' to {comment.file_path}:{comment.line}. "
                        f"Reason: HTTP {response.status_code} ({err_msg})"
                    )
        except Exception as exc:
            failed_details.append(
                f"Could not post '{comment.title}' to {comment.file_path}:{comment.line}. "
                f"Error: {exc}"
            )

    return {
        "created": created,
        "skipped": skipped,
        "failed_details": failed_details,
    }