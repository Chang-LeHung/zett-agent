"""The repository checks that Ruff and mypy do not cover, and the baseline that keeps them from drifting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from static_check import baseline as baseline_file
from static_check.rules import RULES, Violation, check_source
from static_check.scan import python_files, repository_root, scan

ALIAS_WITHOUT_DECLARATION = """
from typing import Any

Payload = dict[str, Any]
Handler = Callable[[int], str]
Names: list[str] = []
Count = 3
Kind = "kind"
"""


def test_a_type_alias_without_a_declaration_is_reported():
    found = check_source("src/zett_agent/somewhere.py", ALIAS_WITHOUT_DECLARATION)

    assert [(violation.rule, violation.line) for violation in found] == [("SC001", 4), ("SC001", 5)]
    assert "Payload" in found[0].message


def test_a_declared_alias_and_an_ordinary_assignment_are_not_reported():
    source = (
        "from typing import TypeAlias\n"
        "Payload: TypeAlias = dict[str, int]\n"
        "Names: list[str] = []\n"
        "Count = 3\n"
        "Kind = 'kind'\n"
        "Value = some_call()\n"
        "Other = int | None\n"
    )

    # `Other` is a union of a builtin and None, so it is a type written without a declaration.
    assert [(violation.rule, violation.line) for violation in check_source("src/a.py", source)] == [("SC001", 7)]


def test_any_is_reported_only_inside_the_providers_package():
    source = "from typing import Any, cast\n\ndef go(value: dict[str, Any]) -> Any:\n    return cast('Any', value)\n"

    assert [violation.rule for violation in check_source("src/zett_agent/providers/one.py", source)] == [
        "SC002",
        "SC002",
        "SC002",
    ]
    assert check_source("src/zett_agent/extensions/two.py", source) == []


def test_any_inside_a_comment_or_a_docstring_is_not_reported():
    source = '''
"""Any is discussed here but not written in a type."""

# Any in a comment
value = 1
'''

    assert check_source("src/zett_agent/providers/three.py", source) == []


def test_an_ignore_marker_silences_one_rule_on_one_line():
    source = "from typing import Any\n\nvalue: Any = None  # static-check: ignore[SC002] third-party payload\nother: Any = None\n"

    found = check_source("src/zett_agent/providers/four.py", source)

    assert [(violation.rule, violation.line) for violation in found] == [("SC002", 4)]


def test_a_private_name_imported_from_another_package_module_is_reported():
    source = "from .other import _helper, public\nfrom typing import _AnnotatedAlias\nfrom . import _module\n"

    found = check_source("src/zett_agent/five.py", source)

    # A private module and a private name from the standard library are not this package's business.
    assert [(violation.rule, violation.line) for violation in found] == [("SC003", 1)]
    assert "_helper" in found[0].message


def test_tests_may_import_private_names():
    source = "from zett_agent.tools.base import _bind_tool_call\n"

    assert check_source("tests/test_x.py", source) == []


def test_every_rule_has_an_id_and_a_docstring():

    assert set(RULES) == {"SC001", "SC002", "SC003"}
    for rule in RULES.values():
        assert rule.__doc__ and rule.__doc__.startswith("SC0")


@pytest.mark.parametrize("name", ["undeclared_type_alias", "explicit_any", "private_name_import"])
def test_a_rule_returns_without_reading_when_it_does_not_apply(name):
    import ast

    from static_check import rules

    empty = ast.parse("value = 1\n")

    assert list(getattr(rules, name)("src/zett_agent/x.py", empty)) == []


# ------------------------------------------------------------------ the baseline


def test_a_baseline_entry_allows_that_many_occurrences_and_no_more():
    violations = [Violation("SC002", "src/a.py", line, "src/a.py", "") for line in (1, 2, 3)]

    unexpected, stale = baseline_file.compare({"SC002": {"src/a.py": 2}}, violations)

    assert [violation.line for violation in unexpected] == [3]
    assert stale == []


def test_an_entry_whose_occurrences_are_gone_is_stale():
    unexpected, stale = baseline_file.compare({"SC002": {"src/a.py": 3}}, [])

    assert unexpected == []
    assert stale == ["SC002: src/a.py (still allows 3)"]


def test_building_a_baseline_counts_each_key():
    violations = [
        Violation("SC002", "src/a.py", 1, "src/a.py", ""),
        Violation("SC002", "src/a.py", 2, "src/a.py", ""),
        Violation("SC003", "src/b.py", 1, "src/b.py: from .c import _d", ""),
    ]

    assert baseline_file.build(violations) == {"SC002": {"src/a.py": 2}, "SC003": {"src/b.py: from .c import _d": 1}}


def test_a_baseline_round_trips_through_the_file(tmp_path: Path):
    built = {"SC002": {"src/a.py": 2}}
    path = tmp_path / "baseline.json"

    baseline_file.dump(built, path)

    assert baseline_file.load(path) == built
    assert list(baseline_file.describe(built)) == ["SC002: 1 entries, 2 occurrences"]


def test_a_missing_baseline_allows_nothing(tmp_path: Path):
    assert baseline_file.load(tmp_path / "absent.json") == {}


@pytest.mark.parametrize(
    "payload",
    [
        [1, 2],
        {"SC002": 3},
        {"SC002": {"src/a.py": 0}},
        {"SC002": {"src/a.py": "2"}},
        {"SC002": {"src/a.py": True}},
    ],
)
def test_a_malformed_baseline_is_refused(tmp_path: Path, payload):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        baseline_file.load(path)


# ------------------------------------------------------------------ the tree itself


def test_the_repository_passes_its_own_rules():
    root = repository_root(Path(__file__))
    unexpected, stale = baseline_file.compare(baseline_file.load(root / baseline_file.BASELINE_PATH), scan(root))

    assert unexpected == []
    assert stale == []


def test_the_scan_reads_the_sources_and_not_the_build_output():
    root = repository_root(Path(__file__))
    paths = {path.relative_to(root).as_posix() for path in python_files(root)}

    assert "src/zett_agent/providers/base.py" in paths
    assert "static_check/rules.py" in paths
    # `tests/test_sandbox_builders.py` carries a directory-shaped word in its name; only real directories count.
    assert "tests/test_sandbox_builders.py" in paths
    assert [path for path in paths if set(Path(path).parts[:-1]) & {"__pycache__", "_build", ".venv"}] == []


def test_a_tree_without_a_pyproject_is_refused(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        repository_root(tmp_path)
