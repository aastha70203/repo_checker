"""Clone and validate GitHub repositories using GitPython."""

from __future__ import annotations

import re
import shutil
import socket
import tempfile
import threading
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import git
from git import RemoteProgress


CLONE_TIMEOUT_SECONDS = 60
MAX_WARN_FILES = 500
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "env", "dist", "build", ".tox"}


class CloneProgress(RemoteProgress):
    """Feeds git clone progress back to a callback."""

    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        super().__init__()
        self.callback = callback

    def update(self, op_code, cur_count, max_count=None, message=""):
        if self.callback and message:
            self.callback(f"Cloning... {message}")


@dataclass
class CloneResult:
    success: bool
    repo_path: Path
    repo_name: str
    github_url: str
    branch: str
    total_files: int
    python_files: int
    file_tree: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    from_cache: bool = False

    def summary(self) -> str:
        if not self.success:
            return f"Clone failed: {self.error}"
        cache_tag = " (from cache)" if self.from_cache else ""
        lines = [
            f"Cloned '{self.repo_name}'{cache_tag}",
            f"   Branch  : {self.branch}",
            f"   Path    : {self.repo_path}",
            f"   Files   : {self.total_files} total, {self.python_files} Python",
        ]
        for warning in self.warnings:
            lines.append(f"   Warning : {warning}")
        return "\n".join(lines)


class RepoCloner:
    """Clone public GitHub repositories to a local temporary directory."""

    GITHUB_REPO_RE = re.compile(
        r"^https?://github\.com/[\w\-\.]+/[\w\-\.]+(\.git)?/?$",
        re.IGNORECASE,
    )
    GITHUB_SUBPATH_RE = re.compile(
        r"^https?://github\.com/([\w\-\.]+/[\w\-\.]+)/(tree|blob|commits|releases|issues|pulls)",
        re.IGNORECASE,
    )
    SSH_RE = re.compile(r"^git@github\.com:([\w\-\.]+/[\w\-\.]+)(\.git)?$")

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(tempfile.mkdtemp(prefix="ai_reviewer_"))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, CloneResult] = {}

    def clone(
        self,
        github_url: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> CloneResult:
        """Clone a GitHub repository with validation and helpful errors."""

        def notify(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        try:
            clean_url, warnings = self._normalize_url(github_url)
        except ValueError as exc:
            return self._fail(github_url, str(exc))

        if clean_url in self._cache:
            notify(f"Using cached clone for {clean_url.split('/')[-1]}")
            cached = self._cache[clean_url]
            cached.from_cache = True
            return cached

        notify("Checking connectivity to GitHub...")
        reachable, net_err = self._check_connectivity()
        if not reachable:
            return self._fail(github_url, net_err)

        repo_name = clean_url.rstrip("/").split("/")[-1].replace(".git", "") or "repo"
        dest = self.base_dir / repo_name
        if dest.exists():
            shutil.rmtree(dest)

        notify(f"Cloning '{repo_name}' from GitHub...")
        repo, git_err = self._clone_with_timeout(clean_url, dest, notify)
        if repo is None:
            return self._fail(github_url, git_err, repo_name=repo_name)

        branch = self._get_branch(repo)
        all_files = self._list_files(dest)
        py_files = [file for file in all_files if file.endswith(".py")]

        if not all_files:
            shutil.rmtree(dest, ignore_errors=True)
            return self._fail(github_url, "Repository is empty; nothing to review.", repo_name=repo_name)

        if not py_files:
            warnings.append(
                "This repo has no Python (.py) files. The agent currently supports Python only."
            )

        if len(all_files) > MAX_WARN_FILES:
            warnings.append(
                f"Large repo detected ({len(all_files)} files). Only the first chunks will be reviewed."
            )

        notify(f"Cloned {len(all_files)} files ({len(py_files)} Python)")

        result = CloneResult(
            success=True,
            repo_path=dest,
            repo_name=repo_name,
            github_url=github_url,
            branch=branch,
            total_files=len(all_files),
            python_files=len(py_files),
            file_tree=self._build_tree(all_files),
            warnings=warnings,
            from_cache=False,
        )
        self._cache[clean_url] = result
        return result

    def cleanup(self) -> None:
        """Delete cloned repositories and clear the session cache."""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir, onexc=self._handle_remove_readonly)
        self._cache.clear()

    def _handle_remove_readonly(self, function, path, excinfo) -> None:
        """Retry Windows cleanup after making Git pack files writable."""
        try:
            Path(path).chmod(stat.S_IWRITE)
            function(path)
        except Exception:
            raise excinfo[1]

    def _normalize_url(self, raw: str) -> tuple[str, List[str]]:
        url = raw.strip()
        warnings: List[str] = []

        ssh_match = self.SSH_RE.match(url)
        if ssh_match:
            owner_repo = ssh_match.group(1)
            raise ValueError(
                "SSH URLs are not supported. Please use the HTTPS URL instead:\n"
                f"  https://github.com/{owner_repo}"
            )

        if re.match(r"https?://(gitlab|bitbucket|codeberg)\.", url, re.I):
            raise ValueError("Only GitHub repositories are supported right now.")

        if re.match(r"^[\w\-\.]+/[\w\-\.]+$", url):
            url = f"https://github.com/{url}"

        if url.lower().startswith("github.com"):
            url = "https://" + url

        sub_match = self.GITHUB_SUBPATH_RE.match(url)
        if sub_match:
            owner_repo = sub_match.group(1)
            url = f"https://github.com/{owner_repo}"
            warnings.append(f"URL pointed to a sub-page; using repo root instead: {url}")

        url = url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]

        if not self.GITHUB_REPO_RE.match(url) and not self.GITHUB_REPO_RE.match(url + ".git"):
            raise ValueError(
                f"'{raw}' does not look like a valid GitHub repository URL.\n"
                "Expected: https://github.com/owner/repo"
            )

        return url, warnings

    def _check_connectivity(self, host: str = "github.com", port: int = 443, timeout: int = 5) -> tuple[bool, str]:
        try:
            socket.setdefaulttimeout(timeout)
            with socket.create_connection((host, port)):
                return True, ""
        except OSError:
            return False, "Cannot reach GitHub. Please check your internet connection and try again."

    def _clone_with_timeout(
        self,
        url: str,
        dest: Path,
        notify: Callable[[str], None],
    ) -> tuple[Optional[git.Repo], str]:
        result_box: Dict[str, object] = {}

        def do_clone() -> None:
            try:
                result_box["repo"] = git.Repo.clone_from(
                    url + ".git",
                    str(dest),
                    depth=1,
                    progress=CloneProgress(notify),
                )
            except git.exc.GitCommandError as exc:
                result_box["error"] = self._parse_git_error(str(exc), url)
            except Exception as exc:
                result_box["error"] = f"Unexpected error during clone: {exc}"

        thread = threading.Thread(target=do_clone, daemon=True)
        thread.start()
        thread.join(timeout=CLONE_TIMEOUT_SECONDS)

        if thread.is_alive():
            shutil.rmtree(dest, ignore_errors=True)
            return None, (
                f"Clone timed out after {CLONE_TIMEOUT_SECONDS}s. "
                "The repository may be too large or the connection too slow."
            )

        if "error" in result_box:
            shutil.rmtree(dest, ignore_errors=True)
            return None, str(result_box["error"])

        return result_box.get("repo"), ""

    def _get_branch(self, repo: git.Repo) -> str:
        try:
            return repo.active_branch.name
        except TypeError:
            return repo.head.commit.hexsha[:7]

    def _list_files(self, path: Path) -> List[str]:
        files: List[str] = []
        for file in path.rglob("*"):
            if not file.is_file():
                continue
            rel_parts = file.relative_to(path).parts
            if any(part in SKIP_DIRS or part.startswith(".") for part in rel_parts):
                continue
            files.append(file.relative_to(path).as_posix())
        return sorted(files)

    def _build_tree(self, files: List[str], max_entries: int = 50) -> List[str]:
        tree = files[:max_entries]
        if len(files) > max_entries:
            tree.append(f"... and {len(files) - max_entries} more files")
        return tree

    def _parse_git_error(self, raw: str, url: str) -> str:
        low = raw.lower()
        if "not found" in low or "repository not found" in low or "exit code(128)" in low:
            return f"Repository not found: {url}\nDouble-check the URL and make sure the repo is public."
        if "authentication failed" in low or "could not read username" in low:
            return "Authentication failed. This repository is likely private."
        if "could not resolve host" in low:
            return "Network error: could not reach GitHub. Check your connection."
        if "already exists and is not an empty directory" in low:
            return "Destination folder already exists; please try again."
        return f"Git error: {raw[:300]}"

    def _fail(self, url: str, error: str, repo_name: str = "") -> CloneResult:
        return CloneResult(
            success=False,
            repo_path=Path(),
            repo_name=repo_name,
            github_url=url,
            branch="",
            total_files=0,
            python_files=0,
            error=error,
        )
