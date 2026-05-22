"""AST parsing and chunking for Python repositories."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Sequence

from agent.models import CodeChunk, CodeSymbol


DEFAULT_MAX_CHUNK_LINES = 180
DEFAULT_MAX_FILE_BYTES = 350_000
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    "node_modules",
}


class PythonASTParser:
    """Extract reviewable Python chunks while preserving line numbers."""

    def __init__(
        self,
        max_chunk_lines: int = DEFAULT_MAX_CHUNK_LINES,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self.max_chunk_lines = max_chunk_lines
        self.max_file_bytes = max_file_bytes

    def iter_python_files(self, repo_path: Path) -> Iterable[Path]:
        for path in sorted(repo_path.rglob("*.py")):
            rel_parts = path.relative_to(repo_path).parts
            if any(part in SKIP_DIRS or part.startswith(".") for part in rel_parts):
                continue
            if path.stat().st_size > self.max_file_bytes:
                continue
            yield path

    def parse_repository(self, repo_path: Path) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        for file_path in self.iter_python_files(repo_path):
            rel_path = file_path.relative_to(repo_path).as_posix()
            chunks.extend(self.parse_file(file_path, rel_path))
        return chunks

    def parse_file(self, file_path: Path, display_path: str | None = None) -> List[CodeChunk]:
        display = display_path or file_path.name
        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if not lines:
            return []

        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return self._window_chunks(display, lines, parse_error=str(exc))

        imports = self._imports(tree)
        symbols = self._symbols(tree)
        top_level_nodes = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]

        if not top_level_nodes:
            return self._window_chunks(display, lines, imports=imports)

        chunks: List[CodeChunk] = []
        used_ranges: List[tuple[int, int]] = []
        for node in top_level_nodes:
            start = getattr(node, "lineno", 1)
            end = getattr(node, "end_lineno", start)
            node_lines = lines[start - 1:end]
            node_symbols = [s for s in symbols if start <= s.start_line <= end]
            if len(node_lines) > self.max_chunk_lines:
                chunks.extend(
                    self._window_chunks(
                        display,
                        node_lines,
                        start_offset=start,
                        imports=imports,
                        symbols=node_symbols,
                    )
                )
            else:
                chunks.append(
                    CodeChunk(
                        file_path=display,
                        language="python",
                        start_line=start,
                        end_line=end,
                        code="\n".join(node_lines),
                        symbols=node_symbols,
                        imports=imports,
                    )
                )
            used_ranges.append((start, end))

        module_prelude = self._module_prelude(lines, used_ranges)
        if module_prelude.strip():
            end_line = len(module_prelude.splitlines())
            chunks.insert(
                0,
                CodeChunk(
                    file_path=display,
                    language="python",
                    start_line=1,
                    end_line=end_line,
                    code=module_prelude,
                    imports=imports,
                    symbols=[],
                ),
            )

        return chunks

    def _window_chunks(
        self,
        display: str,
        lines: Sequence[str],
        start_offset: int = 1,
        imports: List[str] | None = None,
        symbols: List[CodeSymbol] | None = None,
        parse_error: str | None = None,
    ) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        imports = imports or []
        symbols = symbols or []
        for idx in range(0, len(lines), self.max_chunk_lines):
            chunk_lines = list(lines[idx:idx + self.max_chunk_lines])
            start = start_offset + idx
            end = start + len(chunk_lines) - 1
            chunks.append(
                CodeChunk(
                    file_path=display,
                    language="python",
                    start_line=start,
                    end_line=end,
                    code="\n".join(chunk_lines),
                    symbols=[s for s in symbols if start <= s.start_line <= end],
                    imports=imports,
                    parse_error=parse_error,
                )
            )
        return chunks

    def _symbols(self, tree: ast.AST) -> List[CodeSymbol]:
        symbols: List[CodeSymbol] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append(CodeSymbol(node.name, "class", node.lineno, node.end_lineno or node.lineno))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
                symbols.append(CodeSymbol(node.name, kind, node.lineno, node.end_lineno or node.lineno))
        return sorted(symbols, key=lambda item: (item.start_line, item.name))

    def _imports(self, tree: ast.AST) -> List[str]:
        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append(module if node.level == 0 else "." * node.level + module)
        return sorted(set(imports))

    def _module_prelude(self, lines: Sequence[str], used_ranges: Sequence[tuple[int, int]]) -> str:
        if not used_ranges:
            return "\n".join(lines[: self.max_chunk_lines])
        first_symbol_line = min(start for start, _ in used_ranges)
        return "\n".join(lines[: max(0, first_symbol_line - 1)]).strip()
