"""Command line: ``python -m static_check`` checks the tree, ``--update-baseline`` records what it finds."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import baseline as baseline_file
from .rules import RULES
from .scan import repository_root, scan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="static_check", description="Checks Ruff and mypy do not cover.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite the baseline to allow exactly the violations that exist now",
    )
    parser.add_argument("--root", type=Path, default=None, help="repository root; found from the working directory")
    arguments = parser.parse_args(argv)

    root = repository_root(arguments.root)
    path = root / baseline_file.BASELINE_PATH
    violations = scan(root)

    if arguments.update_baseline:
        baseline_file.dump(baseline_file.build(violations), path)
        for line in baseline_file.describe(baseline_file.load(path)):
            print(line)
        return 0

    unexpected, stale = baseline_file.compare(baseline_file.load(path), violations)
    for violation in unexpected:
        print(f"{violation.path}:{violation.line}: {violation.rule} {violation.message}")
    for entry in stale:
        print(f"{baseline_file.BASELINE_PATH}: stale entry {entry}; run `python -m static_check --update-baseline`")
    if unexpected or stale:
        print(f"static_check: {len(unexpected)} violations, {len(stale)} stale baseline entries", file=sys.stderr)
        return 1
    print(f"static_check: {len(RULES)} rules, no new violations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
