"""Which private methods of the ``Agent`` mixins are seams between them, and which are one mixin's own.

A mixin's private method is either a helper only its own file uses or something other mixins build on.
``@mixin_api`` marks the second kind. The source is read, not run, so these rules hold for code no test
reaches:

* a private method that another file calls carries the mark, and a marked one is called from another file;
* a marked method that another *mixin* calls is declared on ``AgentContract``, and a private method
  declared there is marked and called by a mixin;
* no two mixins define a method of the same name, because they share one namespace on ``Agent``.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

from zett_agent._agent.contract import MIXIN_API_MARK, AgentContract
from zett_agent.agent import Agent

SOURCE = Path(__file__).resolve().parents[1] / "src" / "zett_agent"
AGENT = "agent"


def read_sources() -> dict[str, str]:
    """The ``Agent`` file and every mixin file, by short name."""
    sources = {path.stem: path.read_text() for path in sorted((SOURCE / "_agent").glob("*.py"))}
    sources.pop("__init__", None)
    sources[AGENT] = (SOURCE / "agent.py").read_text()
    return sources


def is_marked(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(isinstance(item, ast.Name) and item.id == "mixin_api" for item in node.decorator_list)


def find_problems(sources: Mapping[str, str]) -> list[str]:
    """Every way the files break the rules in the module docstring, one sentence each."""
    trees = {name: ast.parse(text) for name, text in sources.items()}
    defined: dict[str, tuple[str, bool]] = {}  # method -> (file, marked)
    problems: list[str] = []
    for name, tree in trees.items():
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            if name == AGENT and cls.name != "Agent":
                continue
            for node in cls.body:
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) or node.name.startswith("__"):
                    continue
                if node.name in defined and defined[node.name][0] != name:
                    problems.append(f"{node.name} is defined in both {defined[node.name][0]} and {name}")
                defined[node.name] = (name, is_marked(node))

    users: dict[str, set[str]] = {}  # method -> the files that mention it
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in defined:
                users.setdefault(node.attr, set()).add(name)

    contract = contract_methods(trees)
    for method, (owner, marked) in sorted(defined.items()):
        if not method.startswith("_"):
            continue
        elsewhere = users.get(method, set()) - {owner}
        mixins_elsewhere = elsewhere - {AGENT}
        if elsewhere and not marked:
            problems.append(f"{method} is used by {sorted(elsewhere)} but is not marked @mixin_api")
        if marked and not elsewhere:
            problems.append(f"{method} is marked @mixin_api but nothing outside {owner} uses it")
        if marked and mixins_elsewhere and method not in contract:
            problems.append(f"{method} is used by {sorted(mixins_elsewhere)} but AgentContract does not declare it")
        if method in contract and not (marked and mixins_elsewhere):
            problems.append(f"{method} is declared on AgentContract but no other mixin uses it")
    problems.extend(
        f"{method} is declared on AgentContract but no mixin defines it" for method in sorted(contract - defined.keys())
    )
    return problems


def contract_methods(trees: Mapping[str, ast.Module]) -> set[str]:
    """The private methods ``AgentContract`` declares, from the source of ``contract.py``."""
    tree = trees.get("contract")
    if tree is None:
        return set()
    return {
        node.name
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "AgentContract"
        for node in ast.walk(cls)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("_")
    }


def test_the_private_methods_are_split_into_seams_and_a_mixins_own():
    assert find_problems(read_sources()) == []


def test_the_mark_reaches_the_methods_at_runtime_and_nowhere_else():
    """The mark is for readers and for this test; a method is marked in the source exactly when it is at runtime."""
    in_source: set[str] = set()
    for name, text in read_sources().items():
        for cls in (node for node in ast.parse(text).body if isinstance(node, ast.ClassDef)):
            if name == AGENT and cls.name != "Agent":
                continue
            in_source |= {
                node.name
                for node in cls.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and is_marked(node)
            }
    at_runtime = {
        name
        for klass in Agent.__mro__
        for name in vars(klass)
        if getattr(getattr(klass, name), MIXIN_API_MARK, False) is True
    }

    assert in_source and in_source == at_runtime


def test_the_contract_methods_are_all_marked_on_agent():
    declared = contract_methods({"contract": ast.parse((SOURCE / "_agent" / "contract.py").read_text())})

    assert declared
    assert all(getattr(getattr(Agent, name), MIXIN_API_MARK, False) for name in declared)
    assert not [name for name in vars(AgentContract) if name in declared]


# ----------------------------------------------------- the checker catches what it should

MIXIN_A = """
class A:
    @mixin_api
    async def _shared(self): ...
    async def _own(self): await self._own_helper()
    async def _own_helper(self): ...
    async def _leaked(self): ...
    @mixin_api
    async def _unused(self): ...
"""
MIXIN_B = """
class B:
    async def go(self):
        await self._shared()
        await self._leaked()
"""
CONTRACT = """
class AgentContract:
    if TYPE_CHECKING:
        async def _shared(self): ...
        async def _stale(self): ...
"""


def test_the_checker_reports_each_rule_it_enforces():
    problems = find_problems({"a": MIXIN_A, "b": MIXIN_B, "contract": CONTRACT})

    assert sorted(problems) == [
        "_leaked is used by ['b'] but is not marked @mixin_api",
        "_stale is declared on AgentContract but no mixin defines it",
        "_unused is marked @mixin_api but nothing outside a uses it",
    ]


def test_two_mixins_that_define_one_name_are_reported():
    first = "class A:\n    async def _own(self): ...\n"
    second = "class B:\n    async def _own(self): ...\n"

    assert find_problems({"a": first, "b": second}) == ["_own is defined in both a and b"]


def test_a_shared_method_the_contract_does_not_declare_is_reported():
    problems = find_problems({"a": MIXIN_A, "b": MIXIN_B})

    assert "_shared is used by ['b'] but AgentContract does not declare it" in problems
