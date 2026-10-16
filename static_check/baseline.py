"""The record of violations the code already has, which may shrink but not grow.

A rule that cannot be satisfied everywhere today still has to be enforced, or it is advice. The baseline
holds the allowed violations keyed by rule and by the identity of the thing allowed, so a new occurrence is
reported while an old one is not. Lowering a count is how progress is recorded; the tool also reports
entries that are no longer needed, so the file cannot drift away from the code.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from .rules import Violation

BASELINE_PATH = Path("static_check/baseline.json")


def load(path: Path) -> dict[str, dict[str, int]]:
    """Read the baseline, rejecting a file that is not ``{rule: {key: count}}`` with counts of at least one."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must hold an object of {{rule: {{key: count}}}}")
    loaded: dict[str, dict[str, int]] = {}
    for rule, entries in payload.items():
        if not isinstance(rule, str) or not isinstance(entries, dict):
            raise ValueError(f"{path}: {rule!r} must map a rule name to an object of counts")
        counts: dict[str, int] = {}
        for key, count in entries.items():
            if not isinstance(key, str) or not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(f"{path}: {rule}.{key!r} must be a positive integer")
            counts[key] = count
        loaded[rule] = counts
    return loaded


def compare(
    baseline: Mapping[str, Mapping[str, int]], violations: Iterable[Violation]
) -> tuple[list[Violation], list[str]]:
    """Split ``violations`` into the ones the baseline does not cover, and the baseline entries to remove.

    An occurrence consumes one from its key's budget, so a file allowed three ``Any`` uses reports a fourth.
    An entry whose occurrences all disappeared is stale and has to be dropped, which keeps the ratchet tight.
    """
    budget = {rule: Counter(entries) for rule, entries in baseline.items()}
    unexpected: list[Violation] = []
    for violation in violations:
        remaining = budget.setdefault(violation.rule, Counter()).get(violation.key, 0)
        if remaining <= 0:
            unexpected.append(violation)
        else:
            budget[violation.rule][violation.key] = remaining - 1
    stale = [
        f"{rule}: {key} (still allows {count})"
        for rule, entries in sorted(budget.items())
        for key, count in sorted(entries.items())
        if count > 0
    ]
    return unexpected, stale


def build(violations: Iterable[Violation]) -> dict[str, dict[str, int]]:
    """The baseline that exactly allows ``violations``, sorted so the file is stable in a diff."""
    counted: dict[str, Counter[str]] = {}
    for violation in violations:
        counted.setdefault(violation.rule, Counter())[violation.key] += 1
    return {rule: dict(sorted(entries.items())) for rule, entries in sorted(counted.items())}


def dump(baseline: Mapping[str, Mapping[str, int]], path: Path) -> None:
    """Write ``baseline`` as the file the tool reads, sorted so a change shows up as a small diff."""
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def describe(baseline: Mapping[str, Mapping[str, int]]) -> Iterator[str]:
    """One line per rule, saying how many entries and how many occurrences it allows."""
    for rule, entries in sorted(baseline.items()):
        yield f"{rule}: {len(entries)} entries, {sum(entries.values())} occurrences"
