"""Tool snippets must stay readable one-line examples."""

import ast
import re
from pathlib import Path

import zett_agent

_CALL_START = re.compile(r"^\s*[a-z_][a-z0-9_]*\(")


def _broken_examples(source: str) -> list[str]:
    """Return snippet examples that a docstring escape split across lines.

    A single ``\\n`` inside a docstring is a real newline, so an example such as
    ``write_file(content="...\\n...")`` silently renders as a broken call.
    """
    broken: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        document = ast.get_docstring(node, clean=False) or ""
        if "Snippet:" not in document:
            continue
        section = document.split("Snippet:", 1)[1].split("Guidelines:", 1)[0]
        broken.extend(
            f"{node.name}: {line.strip()}"
            for line in section.splitlines()
            if _CALL_START.match(line) and not line.rstrip().endswith(")")
        )
    return broken


def test_every_tool_snippet_example_stays_on_one_line() -> None:
    package = Path(zett_agent.__file__).parent
    broken = [
        f"{module.relative_to(package)} · {example}"
        for module in sorted(package.rglob("*.py"))
        for example in _broken_examples(module.read_text(encoding="utf-8"))
    ]

    assert not broken, f"Tool snippets must escape newlines as '\\\\n' so the example renders on one line: {broken}"
