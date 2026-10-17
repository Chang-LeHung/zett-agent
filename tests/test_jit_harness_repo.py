"""The repository: files on disk, imported through a finder and loader, with ``harness`` in globals."""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from jit_support import HEADER, put_tool, put_workflow, repo_dir, seed_repo, write_module
from zett_agent.agent import Agent, AgentRunConfig, AgentRunContext, AgentState
from zett_agent.jit_harness import JITHarness, JITHarnessExtension
from zett_agent.jit_harness.pages import text_page
from zett_agent.jit_harness.repo import (
    MANAGEMENT_NAMES,
    SKELETON,
    parse_workflow,
    session_directory_name,
    validate_repo_name,
)
from zett_agent.messages import AssistantMessage, SystemMessage, ToolCall
from zett_agent.model import ModelEvent, ModelResponse, RetryOptions
from zett_agent.tools.base import tool

TWICE = dedent('''\
    @tool(guidelines="Use for composed integer addition.")
    async def twice(value: int) -> int:
        """Add the same integer twice."""
        return await harness.call_tool("add", left=value, right=value)
''')


@tool(guidelines="Use for exact integer addition.")
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


def jit_modules():
    return [name for name in sys.modules if name.startswith("zett_jit_")]


def jit_finders():
    return [finder for finder in sys.meta_path if type(finder).__name__ == "RepoFinder"]


@pytest.fixture(autouse=True)
def no_residue():
    """Every test must leave the interpreter's import state as it found it."""
    before = (set(jit_modules()), len(jit_finders()))
    yield
    assert (set(jit_modules()), len(jit_finders())) == before


def test_create_repo_makes_a_package_and_is_idempotent(tmp_path):
    harness = JITHarness(tmp_path / "session", tools=[add])
    assert harness.repo is None and harness.repo_path is None
    assert not (tmp_path / "session").exists()  # Nothing is written before a repository exists.
    created = harness.create_repo("kit")
    path = tmp_path / "session" / "kit"
    assert created == {"repo": "kit", "path": str(path), "status": "created", "tools": []}
    assert (path / "__init__.py").read_text() == SKELETON
    assert harness.repo == "kit" and harness.repo_path == path
    (path / "extra.py").write_text("# the model's work\n")
    again = harness.create_repo("kit")
    assert again["status"] == "unloaded" and (path / "extra.py").exists()
    with pytest.raises(ValueError, match="already has the repository 'kit'"):
        harness.create_repo("other")


@pytest.mark.parametrize("name", ["", "-lead", "has space", "a/b", "..", ".hidden", "x" * 65, "üml"])
def test_repository_names_are_safe_directory_names(name, tmp_path):
    with pytest.raises(ValueError, match="Repository names"):
        JITHarness(tmp_path).create_repo(name)
    with pytest.raises(ValueError, match="Repository names"):
        validate_repo_name(name)
    assert list(tmp_path.iterdir()) == []


def test_load_requires_a_repository(tmp_path):
    harness = JITHarness(tmp_path)
    for operation in (harness.load_repo, harness.unload_repo):
        with pytest.raises(ValueError, match="repo_create"):
            operation()


async def test_load_registers_tools_from_every_module_and_composes_through_the_harness(tmp_path):
    harness = JITHarness(tmp_path, tools=[add])
    repo = repo_dir(harness)
    write_module(harness, "steps", "def double(value):\n    return value * 2\n", header=False)
    write_module(
        harness,
        "pipeline",
        'from .steps import double\n\n@tool(guidelines="Double then add.")\n'
        'async def doubled_sum(value: int) -> int:\n    """Double, then add the value to itself."""\n'
        '    return double(await harness.call_tool("add", left=value, right=value))\n',
    )
    (repo / "nested").mkdir()
    write_module(harness, "nested/__init__", TWICE)
    result = harness.load_repo()
    assert result["status"] == "loaded" and result["tools"] == ["doubled_sum", "twice"]
    assert await harness.call_tool("doubled_sum", value=3) == 12
    assert await harness.call_tool("twice", value=4) == 8
    assert harness.describe("twice")["origin"] == "python"
    harness.close()


async def test_a_module_may_define_several_tools_and_imported_tools_are_not_registered(tmp_path):
    harness = JITHarness(tmp_path, tools=[add])
    put_tool(
        harness,
        "from zett_agent.tools.coding import read_file\n"
        '@tool(guidelines="One.")\ndef one() -> int:\n    """One."""\n    return 1\n\n'
        '@tool(guidelines="Two.")\ndef two() -> int:\n    """Two."""\n    return 2\n',
    )
    assert harness.load_repo()["tools"] == ["one", "two"]  # read_file is imported, not defined here.
    harness.close()


async def test_a_tool_reexported_by_the_package_root_is_registered_once(tmp_path):
    harness = JITHarness(tmp_path)
    write_module(harness, "impl", '@tool(guidelines="One.")\ndef one() -> int:\n    """One."""\n    return 1\n')
    (repo_dir(harness) / "__init__.py").write_text("from .impl import one\n")
    assert harness.load_repo()["tools"] == ["one"]
    harness.close()


def test_harness_is_a_module_global_only(tmp_path):
    harness = JITHarness(tmp_path)
    put_tool(
        harness,
        "seen = harness\nbuiltin = hasattr(__import__('builtins'), 'harness')\n"
        '@tool(guidelines="Probe.")\ndef probe() -> str:\n    """Probe."""\n    return "ok"\n',
    )
    module = next(sys.modules[name] for name in jit_modules() if name.endswith(".tool"))
    assert module.seen is harness and module.__dict__["harness"] is harness
    assert module.builtin is False and not hasattr(builtins, "harness")
    assert "harness" not in vars(sys.modules["zett_agent.jit_harness.repo"])
    harness.close()


async def test_every_module_sees_the_harness_before_its_body_runs(tmp_path):
    harness = JITHarness(tmp_path)
    write_module(harness, "early", "captured = harness\n", header=False)
    (repo_dir(harness) / "__init__.py").write_text("captured = harness\n")
    harness.load_repo()
    captured = {name: sys.modules[name].captured for name in jit_modules()}
    assert len(captured) == 2 and all(value is harness for value in captured.values())
    harness.close()


def test_tracebacks_point_at_real_files_and_the_reload_keeps_the_previous_version(tmp_path):
    harness = JITHarness(tmp_path)
    put_tool(harness, '@tool(guidelines="Old.")\ndef old() -> int:\n    """Old."""\n    return 1\n', "first")
    before = set(jit_modules())
    write_module(harness, "second", "\nvalue = 1 / 0\n", header=False)
    with pytest.raises(ValueError) as caught:
        harness.load_repo()
    message = str(caught.value)
    assert "second.py:2 in <module>: value = 1 / 0" in message
    assert "ZeroDivisionError: division by zero" in message
    assert str(tmp_path) not in message  # Paths are repository-relative.
    assert harness.describe("old")["status"] == "loaded"
    assert set(jit_modules()) == before  # The failed attempt left no modules behind.
    (repo_dir(harness) / "second.py").write_text("def broken(:\n")
    with pytest.raises(ValueError, match=r"second.py:1: def broken\(:\n.*SyntaxError"):
        harness.load_repo()
    harness.close()


async def test_reload_replaces_the_whole_set_and_removed_tools_leave_the_request(tmp_path):
    harness = JITHarness(tmp_path)
    put_tool(harness, '@tool(guidelines="A.")\ndef a() -> int:\n    """A."""\n    return 1\n', "a")
    put_tool(harness, '@tool(guidelines="B.")\ndef b() -> int:\n    """B."""\n    return 2\n', "b")
    assert sorted(harness.load_repo()["tools"]) == ["a", "b"]
    (repo_dir(harness) / "a.py").unlink()
    write_module(harness, "b", '@tool(guidelines="B.")\ndef b() -> int:\n    """B."""\n    return 3\n')
    assert harness.load_repo()["tools"] == ["b"]
    assert await harness.call_tool("b") == 3
    with pytest.raises(ValueError, match="Unknown tool"):
        await harness.call_tool("a")
    harness.close()


async def test_the_new_version_answers_after_a_reload(tmp_path):
    harness = JITHarness(tmp_path)
    template = '@tool(guidelines="Pick.")\ndef pick() -> int:\n    """Pick."""\n    return {}\n'
    for value in (1, 2, 3):  # Same length, same second: a .pyc keyed on mtime and size would be stale.
        put_tool(harness, template.format(value), "pick")
        assert await harness.call_tool("pick") == value
    assert not list(repo_dir(harness).rglob("__pycache__"))
    harness.close()


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ('@tool(name="repo_load", guidelines="x")\ndef a() -> int:\n    """A."""\n    return 1\n', "reserved"),
        ('@tool(name="has space", guidelines="x")\ndef a() -> int:\n    """A."""\n    return 1\n', "Tool names"),
        ('@tool(name="add", guidelines="x")\ndef a() -> int:\n    """A."""\n    return 1\n', "application tool"),
        ('@tool(deferred=True, guidelines="x")\ndef a() -> int:\n    """A."""\n    return 1\n', "non-deferred"),
    ],
)
def test_invalid_names_and_kinds_are_rejected_without_registering_anything(tmp_path, source, message):
    harness = JITHarness(tmp_path, tools=[add])
    write_module(harness, "bad", source)
    with pytest.raises(ValueError, match=message):
        harness.load_repo()
    assert harness.list_tools() == [harness.describe("add")]
    harness.close()


def test_two_different_tools_may_not_share_a_name(tmp_path):
    harness = JITHarness(tmp_path)
    for module in ("one", "two"):
        write_module(
            harness, module, '@tool(name="same", guidelines="x")\ndef f() -> int:\n    """F."""\n    return 1\n'
        )
    with pytest.raises(ValueError, match="Two tools are named 'same'"):
        harness.load_repo()
    harness.close()


def test_unload_hides_every_tool_and_load_restores_them_from_disk(tmp_path):
    harness = JITHarness(tmp_path)
    put_tool(harness, '@tool(guidelines="A.")\ndef a() -> int:\n    """A."""\n    return 1\n', "a")
    assert harness.unload_repo() == {**harness.load_repo(), "status": "unloaded"}
    assert harness.describe("a")["status"] == "loaded"  # load_repo above reloaded it.
    harness.unload_repo()
    assert harness.describe("a")["status"] == "unloaded"
    assert harness.create_repo("repo")["status"] == "unloaded"
    harness.load_repo()
    assert harness.describe("a")["status"] == "loaded"
    harness.close()


async def test_lazy_sibling_imports_work_while_the_repository_is_loaded_and_close_removes_everything(tmp_path):
    harness = JITHarness(tmp_path)
    write_module(harness, "late", "VALUE = 7\n", header=False)
    put_tool(
        harness,
        '@tool(guidelines="Late.")\ndef late_value() -> int:\n    """Import a sibling when called."""\n'
        "    from .late import VALUE\n    return VALUE\n",
        "user",
    )
    assert await harness.call_tool("late_value") == 7
    assert jit_modules() and jit_finders()
    harness.close()
    assert not jit_modules() and not jit_finders()
    harness.close()  # Closing twice is harmless.


def test_a_reload_closes_the_previous_package(tmp_path):
    harness = JITHarness(tmp_path)
    put_tool(harness, '@tool(guidelines="A.")\ndef a() -> int:\n    """A."""\n    return 1\n', "a")
    first = set(jit_modules())
    harness.load_repo()
    assert len(jit_finders()) == 1 and not (first & set(jit_modules()))
    harness.close()


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("w", "no front matter", "must start with a front matter"),
        ("w", "---\ndescription: x\n", "never closed"),
        ("w", "---\n---\nbody", "needs a description"),
        ("w", "---\ndescription: x\n---\n \n", "no content"),
        ("w", "---\ntitle: x\n---\nbody", "supports only 'description"),
        ("w", f"---\ndescription: {'x' * 513}\n---\nbody", "1–512"),
        ("repo_load", "---\ndescription: x\n---\nbody", "reserved"),
    ],
)
def test_workflow_front_matter_is_strict(name, content, message):
    with pytest.raises(ValueError, match=message):
        parse_workflow(name, content)


def test_workflow_front_matter_accepts_quotes_and_blank_lines():
    workflow = parse_workflow("w", '---\n\ndescription: "Use when needed."\n---\n\n# Title\nstep\n')
    assert (workflow.description, workflow.content, workflow.path) == (
        "Use when needed.",
        "# Title\nstep",
        "workflows/w.md",
    )


async def test_workflows_register_as_readers_beside_tools_and_page_exactly(tmp_path):
    harness = JITHarness(tmp_path)
    body = "# Steps\n" + "line\n" * 50
    put_tool(harness, '@tool(guidelines="A.")\ndef a() -> int:\n    """A."""\n    return 1\n', "a", reload=False)
    put_workflow(harness, "triage", "Use to triage failures.", body)
    assert harness.describe("triage")["origin"] == "text"
    assert harness.describe("triage")["description"] == "Read workflow: Use to triage failures."
    assert harness.load_repo()["tools"] == ["a", "triage"]
    page = await harness.call_tool("triage", max_chars=20)
    stored = body.strip("\n")  # Surrounding blank lines are not part of the workflow.
    assert page["path"] == "workflows/triage.md" and page["content"] == stored[:20] and page["next_offset"] == 20
    rest = await harness.call_tool("triage", offset=20, max_chars=4000)
    assert page["content"] + rest["content"] == stored and rest["next_offset"] is None
    harness.close()


def test_a_workflow_and_a_tool_may_not_share_a_name(tmp_path):
    harness = JITHarness(tmp_path)
    put_tool(harness, '@tool(guidelines="A.")\ndef shared() -> int:\n    """A."""\n    return 1\n', "a", reload=False)
    put_workflow(harness, "shared", "Same name.", "body", reload=False)
    with pytest.raises(ValueError, match="share the name 'shared'"):
        harness.load_repo()
    harness.close()


def test_a_broken_workflow_fails_the_load_before_anything_is_imported(tmp_path):
    harness = JITHarness(tmp_path)
    put_tool(harness, '@tool(guidelines="A.")\ndef a() -> int:\n    """A."""\n    return 1\n', "a", reload=False)
    (repo_dir(harness) / "workflows").mkdir()
    (repo_dir(harness) / "workflows" / "bad.md").write_text("nothing")
    with pytest.raises(ValueError, match="front matter"):
        harness.load_repo()
    assert not jit_modules()


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("not-importable.py", "not an importable module name"),
        ("pkg-dir/x.py", "not an importable module name"),
    ],
)
def test_module_names_must_be_importable(tmp_path, relative, message):
    harness = JITHarness(tmp_path)
    path = repo_dir(harness) / relative
    path.parent.mkdir(exist_ok=True)
    path.write_text("x = 1\n")
    with pytest.raises(ValueError, match=message):
        harness.load_repo()


def test_a_repository_must_stay_a_package_and_stay_within_limits(tmp_path):
    harness = JITHarness(tmp_path)
    (repo_dir(harness) / "__init__.py").unlink()
    with pytest.raises(ValueError, match="no __init__.py"):
        harness.load_repo()
    (repo_dir(harness) / "__init__.py").write_text("")
    (repo_dir(harness) / "big.py").write_text("#" * 200_001)
    with pytest.raises(ValueError, match="exceeds 200000 bytes"):
        harness.load_repo()
    (repo_dir(harness) / "big.py").unlink()
    for index in range(65):
        (repo_dir(harness) / f"m{index}.py").write_text("")
    with pytest.raises(ValueError, match="at most 64 modules"):
        harness.load_repo()


def test_hidden_cache_and_dot_directories_are_not_imported(tmp_path):
    harness = JITHarness(tmp_path)
    repo = repo_dir(harness)
    for folder in (".git", "__pycache__"):
        (repo / folder).mkdir()
        (repo / folder / "boom.py").write_text("raise RuntimeError('never imported')\n")
    (repo / ".hidden.py").write_text("raise RuntimeError('never imported')\n")
    assert harness.load_repo()["tools"] == []
    harness.close()


def test_namespace_subdirectories_without_init_are_importable(tmp_path):
    harness = JITHarness(tmp_path)
    folder = repo_dir(harness) / "plain"
    folder.mkdir()
    (folder / "inner.py").write_text(
        HEADER + '@tool(guidelines="I.")\ndef inner() -> int:\n    """I."""\n    return 5\n'
    )
    assert harness.load_repo()["tools"] == ["inner"]
    harness.close()


def test_import_time_code_cannot_change_the_harness_or_call_tools(tmp_path):
    harness = JITHarness(tmp_path, tools=[add])
    write_module(harness, "evil", "harness.unload_tool('add')\n", header=False)
    with pytest.raises(ValueError, match="Cannot change the harness while loading"):
        harness.load_repo()
    assert harness.describe("add")["status"] == "loaded"
    harness.close()


def test_baseexception_in_a_module_body_propagates_and_cleans_up(tmp_path):
    harness = JITHarness(tmp_path)
    write_module(harness, "quit", "raise KeyboardInterrupt\n", header=False)
    with pytest.raises(KeyboardInterrupt):
        harness.load_repo()


@pytest.mark.parametrize(
    ("session", "expected"),
    [
        ("plain-session_1.v2", "plain-session_1.v2"),
        ("0189f2c8-aaaa", "0189f2c8-aaaa"),
    ],
)
def test_conventional_session_ids_are_used_as_directory_names(session, expected):
    assert session_directory_name(session) == expected


@pytest.mark.parametrize(
    "session", ["..", ".", "../escape", "a/b", "a\\b", ".hidden", "x" * 300, "", "ünï", "trailing."]
)
def test_other_session_ids_become_distinct_safe_names(session):
    name = session_directory_name(session)
    assert name != session and "/" not in name and "\\" not in name and not name.startswith(".")
    assert 0 < len(name) <= 128 and session_directory_name(session) == name
    assert name != session_directory_name(session + "x")


def test_management_names_are_the_three_repository_tools():
    assert MANAGEMENT_NAMES == {"repo_create", "repo_load", "repo_unload"}


@pytest.mark.parametrize(
    ("offset", "max_chars", "message"),
    [(-1, 10, "offset"), (True, 10, "offset"), (0, 0, "max_chars"), (0, 4001, "max_chars"), (0, True, "max_chars")],
)
def test_text_pages_reject_invalid_bounds(offset, max_chars, message):
    with pytest.raises(ValueError, match=message):
        text_page("w", "abc", offset, max_chars, "workflows/w.md")


def test_a_text_page_shrinks_when_json_escaping_would_exceed_the_budget():
    content = '"' * 4000  # Every quote costs two characters once serialized.
    page = text_page("w", content, 0, 4000, "workflows/w.md")
    assert 0 < len(page["content"]) < 4000 and page["next_offset"] == len(page["content"])
    assert len(json.dumps(page)) <= 8000


def test_a_text_page_refuses_metadata_that_alone_exceeds_the_budget():
    with pytest.raises(ValueError, match="exceeds the response budget"):
        text_page("n" * 9000, "abc", 0, 10, "workflows/w.md")


def test_an_oversized_workflow_file_is_refused(tmp_path):
    harness = JITHarness(tmp_path)
    put_workflow(harness, "big", "Too large.", "x" * 200_001, reload=False)
    with pytest.raises(ValueError, match="workflows/big.md exceeds 200000 bytes"):
        harness.load_repo()


async def test_a_lazy_import_of_a_missing_sibling_fails_like_any_import(tmp_path):
    harness = JITHarness(tmp_path)
    put_tool(
        harness,
        '@tool(guidelines="Late.")\ndef late() -> int:\n    """Import a sibling that does not exist."""\n'
        "    from .absent import VALUE\n    return VALUE\n",
    )
    with pytest.raises(ModuleNotFoundError, match="absent"):
        await harness.call_tool("late")
    harness.close()


async def test_a_reload_removes_deleted_tools_from_the_request_that_is_running(tmp_path):
    harness = JITHarness(tmp_path, tools=[add])
    ctx = AgentRunContext(AgentRunConfig("live"), AgentState(), {"add": add}, ())
    harness.attach(ctx)
    try:
        put_tool(harness, '@tool(guidelines="A.")\ndef gone() -> int:\n    """A."""\n    return 1\n', "gone")
        assert "gone" in ctx.tools
        (repo_dir(harness) / "gone.py").unlink()
        harness.load_repo()
        assert "gone" not in ctx.tools and "add" in ctx.tools
    finally:
        harness.detach(ctx)
        harness.close()


# --- extension and request lifecycle ---------------------------------------------------


class ScriptModel:
    retry = RetryOptions(max_retries=0)

    def __init__(self, *batches):
        self.batches = batches
        self.requests = []

    async def stream(self, request):
        index = len(self.requests)
        self.requests.append(request)
        message = (
            AssistantMessage(tool_calls=self.batches[index])
            if index < len(self.batches)
            else AssistantMessage(content="done")
        )
        yield ModelEvent.completed(ModelResponse(message))


def test_the_extension_offers_exactly_three_tools_in_serial_mode(tmp_path):
    from zett_agent.tools.base import ToolExecutionMode

    management = JITHarnessExtension(tmp_path)._management_tools(JITHarness(tmp_path))
    assert [item.name for item in management] == ["repo_create", "repo_load", "repo_unload"]
    assert all(item.execution_mode is ToolExecutionMode.SERIAL and item.guidelines for item in management)
    assert {item.name for item in management} == MANAGEMENT_NAMES


def test_each_session_gets_its_own_directory_under_the_root(tmp_path):
    jit = JITHarnessExtension(tmp_path / "jit")
    assert jit.session_directory("alpha") == tmp_path / "jit" / "alpha"
    assert jit.session_directory("../x").parent == tmp_path / "jit"
    assert not (tmp_path / "jit").exists()


@pytest.mark.parametrize("value", [0, -1])
def test_extension_validates_its_bounds(tmp_path, value):
    with pytest.raises(ValueError, match="max_call_depth"):
        JITHarnessExtension(tmp_path, max_call_depth=value)
    with pytest.raises(ValueError, match="manage_base_tools"):
        JITHarnessExtension(tmp_path, manage_base_tools="yes")  # type: ignore[arg-type]  # deliberately invalid


@tool(guidelines="Write one file for the test.")
def put(path: str, content: str) -> str:
    """Write a file."""
    Path(path).write_text(content, encoding="utf-8")
    return "ok"


async def test_a_model_creates_writes_loads_and_calls_through_the_whole_loop(tmp_path):
    root = tmp_path / "jit"
    target = root / "loop" / "kit"
    module = HEADER + TWICE
    model = ScriptModel(
        (ToolCall("c", "repo_create", {"repo": "kit"}),),
        (ToolCall("w", "put", {"path": str(target / "twice.py"), "content": module}),),
        (ToolCall("l", "repo_load", {}),),
        (ToolCall("t", "twice", {"value": 21}),),
    )
    jit = JITHarnessExtension(root)
    agent = await Agent.create(model, tools=[add, put], extensions=[jit])
    result = await agent.run("go", config=AgentRunConfig("loop"))
    assert result.content == "done"
    names = [{definition.name for definition in request.tools} for request in model.requests]
    assert {"repo_create", "repo_load", "repo_unload"} <= names[0] and "twice" not in names[0]
    assert "twice" in names[3]
    messages = [m for m in agent.state.messages if getattr(m, "name", None) in {"repo_create", "repo_load", "twice"}]
    assert [m.name for m in messages] == ["repo_create", "repo_load", "twice"]
    assert str(target) in messages[0].content and '"twice"' in messages[1].content and messages[2].content == "42"
    # The next request now names the repository, so a resumed session finds its path.
    guidance = next(
        m.content for m in model.requests[3].messages if isinstance(m, SystemMessage) and "Your repository" in m.content
    )
    assert f"`kit` at `{target}`" in guidance
    jit.forget_session("loop")
    assert not jit_modules() and not jit_finders()


async def test_a_resumed_session_adopts_its_repository_without_the_model_asking(tmp_path):
    seed_repo(tmp_path, "resumed", {"twice": TWICE}, name="kit")
    model = ScriptModel((ToolCall("t", "twice", {"value": 5}),))
    jit = JITHarnessExtension(tmp_path)
    agent = await Agent.create(model, tools=[add], extensions=[jit])
    await agent.run("go", config=AgentRunConfig("resumed"))
    assert "twice" in {definition.name for definition in model.requests[0].tools}
    assert jit.get_harness("resumed").repo == "kit"
    assert any(getattr(m, "name", None) == "twice" and m.content == "10" for m in agent.state.messages)
    jit.forget_session("resumed")


async def test_a_broken_repository_does_not_stop_the_session_and_repo_load_reports_it(tmp_path):
    directory = seed_repo(tmp_path, "broken", {}, name="kit")
    (directory / "oops.py").write_text("raise RuntimeError('half written')\n")
    model = ScriptModel((ToolCall("l", "repo_load", {}),))
    jit = JITHarnessExtension(tmp_path)
    agent = await Agent.create(model, tools=[add], extensions=[jit])
    await agent.run("go", config=AgentRunConfig("broken"))
    reply = next(m for m in agent.state.messages if getattr(m, "name", None) == "repo_load")
    assert reply.success is False and "oops.py:1 in <module>" in reply.content and "half written" in reply.content
    assert jit.get_harness("broken").repo == "kit"  # The model can fix the files and load again.
    jit.forget_session("broken")


def test_several_repositories_in_one_session_directory_are_not_guessed(tmp_path):
    seed_repo(tmp_path, "twins", {"twice": TWICE}, name="one")
    second = tmp_path / "twins" / "two"
    second.mkdir()
    (second / "__init__.py").write_text("")
    harness = JITHarness(tmp_path / "twins", tools=[add])
    harness._adopt_existing()
    assert harness.repo is None


def test_forgetting_a_session_closes_its_modules_but_keeps_its_files(tmp_path):
    harness = JITHarness(tmp_path)
    put_tool(harness, '@tool(guidelines="A.")\ndef a() -> int:\n    """A."""\n    return 1\n', "a")
    files = sorted(path.name for path in repo_dir(harness).iterdir())
    harness.close()
    assert not jit_modules()
    assert sorted(path.name for path in repo_dir(harness).iterdir()) == files
    again = JITHarness(tmp_path)
    again._adopt_existing()
    assert again.repo == "repo" and again.load_repo()["tools"] == ["a"]
    again.close()
