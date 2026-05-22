"""Focused tests for parser, review models, and deterministic reviewer behavior."""

from pathlib import Path

from agent.models import ReviewComment
from agent.output import comments_to_markdown
from agent.parser import PythonASTParser
from agent.reviewer import CodeReviewer


def test_parser_chunks_large_function(tmp_path: Path):
    source = "import os\n\n\ndef giant():\n" + "\n".join(f"    x_{i} = {i}" for i in range(20))
    file_path = tmp_path / "sample.py"
    file_path.write_text(source, encoding="utf-8")

    parser = PythonASTParser(max_chunk_lines=8)
    chunks = parser.parse_file(file_path, "sample.py")

    assert len(chunks) >= 3
    assert chunks[0].file_path == "sample.py"
    assert any("os" in chunk.imports for chunk in chunks)


def test_parser_reports_syntax_error_as_chunk(tmp_path: Path):
    file_path = tmp_path / "broken.py"
    file_path.write_text("def broken(:\n    pass\n", encoding="utf-8")

    chunks = PythonASTParser(max_chunk_lines=20).parse_file(file_path, "broken.py")

    assert len(chunks) == 1
    assert chunks[0].parse_error


def test_reviewer_confidence_and_verify_label():
    comment = ReviewComment(
        file_path="x.py",
        line=1,
        severity="info",
        category="maintainability",
        title="Todo",
        comment="Contains TODO.",
        suggestion="Verify.",
        confidence=55,
    )

    assert comment.verify_label == "verify this"


def test_heuristic_reviewer_finds_security_issue(tmp_path: Path):
    file_path = tmp_path / "unsafe.py"
    file_path.write_text("def run(user_input):\n    return eval(user_input)\n", encoding="utf-8")
    chunk = PythonASTParser().parse_file(file_path, "unsafe.py")[0]

    comments = CodeReviewer(use_llm=False).review_chunk(chunk)

    assert any(c.severity == "critical" and c.category == "security" for c in comments)
    assert all(0 <= c.confidence <= 100 for c in comments)


def test_reviewer_provider_defaults():
    assert CodeReviewer(provider="google").model == "gemini-2.5-flash"
    assert CodeReviewer(provider="openai").model == "gpt-4o-mini"


def test_google_payload_json_parsing():
    reviewer = CodeReviewer(provider="google", use_llm=False)
    payload = reviewer._parse_google_payload(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"comments":[{"line":2,"severity":"low","category":"style","title":"Name","comment":"Tighten name.","suggestion":"Rename it.","confidence":72}]}'
                            }
                        ]
                    }
                }
            ]
        }
    )

    assert payload["comments"][0]["confidence"] == 72


def test_markdown_marks_low_confidence():
    comment = ReviewComment(
        file_path="x.py",
        line=5,
        severity="info",
        category="maintainability",
        title="Check TODO",
        comment="A TODO may need follow-up.",
        suggestion="Verify whether it is still needed.",
        confidence=40,
    )

    markdown = comments_to_markdown([comment])

    assert "verify this" in markdown
    assert "Confidence: 40%" in markdown
