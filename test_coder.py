"""Focused tests for parser, review models, and deterministic reviewer behavior."""

import sys
from types import SimpleNamespace
from pathlib import Path

from agent.models import ReviewComment
from agent.output import comments_to_markdown
from agent.parser import PythonASTParser
from agent.pipeline import CodeReviewAgent
from agent.reviewer import CodeReviewer
from agent.schema import validate_review_payload


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
    assert CodeReviewer(provider="google").provider_configured() is False


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


def test_schema_validation_normalizes_bad_model_output(tmp_path: Path):
    file_path = tmp_path / "sample.py"
    file_path.write_text("print('hello')\n", encoding="utf-8")
    chunk = PythonASTParser().parse_file(file_path, "sample.py")[0]
    comments = validate_review_payload(
        {
            "comments": [
                {
                    "line": 999,
                    "severity": "urgent",
                    "category": "unknown",
                    "title": "x" * 200,
                    "comment": "Useful comment",
                    "suggestion": "",
                    "confidence": 150,
                },
                "not a dict",
                {"comment": ""},
            ]
        },
        chunk,
        source="test",
    )

    assert len(comments) == 1
    assert comments[0].line == chunk.end_line
    assert comments[0].severity == "info"
    assert comments[0].category == "maintainability"
    assert comments[0].confidence == 100


def test_google_review_uses_mocked_response(monkeypatch, tmp_path: Path):
    file_path = tmp_path / "sample.py"
    file_path.write_text("def f():\n    return 1\n", encoding="utf-8")
    chunk = PythonASTParser().parse_file(file_path, "sample.py")[0]

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"comments":[{"line":1,"severity":"low","category":"style","title":"Small","comment":"Mocked Gemini comment.","suggestion":"Keep it tidy.","confidence":80}]}'
                                }
                            ]
                        }
                    }
                ]
            }

    def fake_post(*args, **kwargs):
        assert "x-goog-api-key" in kwargs["headers"]
        assert kwargs["json"]["generationConfig"]["responseMimeType"] == "application/json"
        return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=fake_post))

    comments = CodeReviewer(provider="google", model="gemini-2.5-flash").review_chunk(chunk)

    assert comments[0].source == "google-ai-studio"
    assert comments[0].confidence == 80


def test_openai_review_uses_mocked_response(monkeypatch, tmp_path: Path):
    file_path = tmp_path / "sample.py"
    file_path.write_text("def f():\n    return 1\n", encoding="utf-8")
    chunk = PythonASTParser().parse_file(file_path, "sample.py")[0]

    message = SimpleNamespace(
        content='{"comments":[{"line":1,"severity":"info","category":"maintainability","title":"Mock","comment":"Mocked OpenAI comment.","suggestion":"No change.","confidence":77}]}'
    )
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice])
    completions = SimpleNamespace(create=lambda **kwargs: response)
    chat = SimpleNamespace(completions=completions)
    fake_openai = SimpleNamespace(OpenAI=lambda: SimpleNamespace(chat=chat))

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    comments = CodeReviewer(provider="openai", model="gpt-4o-mini").review_chunk(chunk)

    assert comments[0].source == "openai"
    assert comments[0].confidence == 77


def test_pipeline_orchestrates_fake_components(tmp_path: Path):
    chunk = SimpleNamespace(
        file_path="sample.py",
        language="python",
        start_line=1,
        end_line=1,
        code="print('hello')",
        symbols=[],
        imports=[],
        parse_error=None,
    )
    comment = ReviewComment(
        file_path="sample.py",
        line=1,
        severity="info",
        category="maintainability",
        title="Mock",
        comment="Pipeline comment.",
        suggestion="No change.",
        confidence=90,
    )

    clone_result = SimpleNamespace(
        success=True,
        repo_path=tmp_path,
        repo_name="demo",
        branch="main",
        warnings=[],
    )

    agent = CodeReviewAgent.__new__(CodeReviewAgent)
    agent.max_chunks = 50
    agent.cloner = SimpleNamespace(clone=lambda repo_url, progress_callback=None: clone_result)
    agent.parser = SimpleNamespace(parse_repository=lambda repo_path: [chunk])
    agent.reviewer = SimpleNamespace(
        use_llm=False,
        provider="google",
        provider_configured=lambda: False,
        review_chunk=lambda code_chunk: [comment],
    )

    run = agent.review_repository("demo/repo")

    assert run.repo_name == "demo"
    assert run.chunks_reviewed == 1
    assert run.comments == [comment]


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
