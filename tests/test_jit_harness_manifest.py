"""A repository declares itself in MANIFEST and may forbid tool names in BAN_TOOLS.

The checker is ordinary file work: it never imports the repository, needs no session,
and answers for a directory no harness has opened. The loader reads the declarations
once, before the import, and judges what the import produced against that snapshot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jit_support import put_tool, repo_dir
from zett_agent.jit_harness import (
    JITHarness,
    RepositoryPolicy,
    SubprocessToolHost,
    inspect_repository,
    read_banned_tools,
    read_manifest,
)
from zett_agent.jit_harness.repo import scan_repo

MANIFEST = 'name = "repo"\nversion = "0.1.0"\ndescription = "Reusable tools."\n'


def _source(name: str, body: str = "return 1") -> str:
    return f'@tool(guidelines="Use it.")\ndef {name}() -> int:\n    """Run."""\n    {body}\n'


@pytest.fixture
def make(tmp_path):
    """Build a repository and close every harness it created."""
    harnesses: list[JITHarness] = []

    def build(manifest: str = MANIFEST, banned: str | None = None, *, host=None):
        harness = JITHarness(tmp_path, host=host)
        harnesses.append(harness)
        path = repo_dir(harness)
        (path / "MANIFEST").write_text(manifest, encoding="utf-8")
        if banned is not None:
            (path / "BAN_TOOLS").write_text(banned, encoding="utf-8")
        return harness, path

    yield build
    for harness in harnesses:
        harness.close()


def test_creating_a_repository_writes_a_manifest_that_passes(tmp_path):
    harness = JITHarness(tmp_path)
    path = repo_dir(harness)

    assert (path / "MANIFEST").is_file()
    policy = inspect_repository(path)
    assert policy.ok and policy.manifest is not None
    assert policy.manifest.name == "repo" and policy.banned == ()
    harness.close()


def test_a_manifest_is_read_into_a_typed_record(make):
    _, path = make('name = "repo"\nversion = "1.2.3"\ndescription = "Tools."\ntools = ["twice"]\n')

    manifest = read_manifest(path)

    assert (manifest.name, manifest.version, manifest.description, manifest.tools) == (
        "repo",
        "1.2.3",
        "Tools.",
        ("twice",),
    )


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("", "needs name"),
        ('name = "repo"\n', "needs version"),
        ('name = "repo"\nversion = "0.1.0"\n', "needs description"),
        ('name = "repo"\nversion = "0.1.0"\ndescription = "d"\ncolour = "red"\n', "unknown key 'colour'"),
        ('name = "other"\nversion = "0.1.0"\ndescription = "d"\n', "must match the directory 'repo'"),
        ('name = "repo"\nversion = "one"\ndescription = "d"\n', "must look like 1 or 1.2.3"),
        ('name = "repo"\nversion = "0.1.0"\ndescription = ""\n', "description must be 1–512 characters"),
        ('name = "repo"\nversion = "0.1.0"\ndescription = "d"\ntools = "twice"\n', "tools must be a list"),
        ('name = "repo"\nversion = "0.1.0"\ndescription = "d"\ntools = []\n', "tools is empty"),
        ('name = "repo"\nversion = "0.1.0"\ndescription = "d"\ntools = [1]\n', "tools must be a list"),
        ('name = "repo"\nversion = "0.1.0"\ndescription = "d"\ntools = ["not a name"]\n', "is not a tool name"),
        ('name = "repo"\nversion = "0.1.0"\ndescription = "d"\ntools = ["a", "a"]\n', "more than once"),
    ],
)
def test_a_manifest_that_breaks_the_rules_names_what_is_wrong(make, manifest, message):
    _, path = make(manifest)

    with pytest.raises(ValueError, match=message):
        read_manifest(path)


def test_a_missing_manifest_says_what_to_write(make):
    _, path = make()
    (path / "MANIFEST").unlink()

    with pytest.raises(ValueError, match="write name = 'repo'"):
        read_manifest(path)


def test_a_manifest_that_is_not_toml_says_so(make):
    _, path = make("name = \n")

    with pytest.raises(ValueError, match="not valid TOML"):
        read_manifest(path)


@pytest.mark.parametrize("label", ["MANIFEST", "BAN_TOOLS"])
def test_a_file_that_is_not_utf8_names_itself(make, label):
    _, path = make(banned="twice\n")
    (path / label).write_bytes(b'name = "repo"\n\xff\n')

    policy = inspect_repository(path)

    assert policy.ok is False
    assert any(f"{label} is not UTF-8" in problem for problem in policy.problems)


def test_a_byte_order_mark_is_tolerated(make):
    _, path = make("\ufeff" + MANIFEST)

    assert read_manifest(path).name == "repo"


def test_a_ban_list_is_one_name_per_line_and_blank_lines_are_ignored(make):
    _, path = make(banned="twice\n\n  other  \n")

    assert read_banned_tools(path) == ("twice", "other")


def test_a_missing_ban_list_bans_nothing(make):
    _, path = make()

    assert read_banned_tools(path) == ()


@pytest.mark.parametrize(
    ("banned", "message"),
    [("not a name\n", "is not a tool name"), ("twice\ntwice\n", "more than once")],
)
def test_a_broken_ban_list_names_what_is_wrong(make, banned, message):
    _, path = make(banned=banned)

    with pytest.raises(ValueError, match=message):
        read_banned_tools(path)


def test_a_banned_tool_may_not_be_defined(make):
    harness, path = make(banned="twice\n")
    put_tool(harness, _source("twice"), "twice", reload=False)

    with pytest.raises(ValueError, match="'twice', which BAN_TOOLS bans"):
        harness.load_repo()
    assert harness.list_tools() == []


def test_a_declared_tool_must_exist(make):
    harness, path = make(MANIFEST + 'tools = ["twice"]\n')
    put_tool(harness, _source("other"), "other", reload=False)

    with pytest.raises(ValueError, match="declares 'twice', which the repository does not define"):
        harness.load_repo()
    assert harness.list_tools() == []


def test_a_tool_outside_the_declaration_is_refused(make):
    harness, path = make(MANIFEST + 'tools = ["twice"]\n')
    put_tool(harness, _source("twice"), "twice", reload=False)
    put_tool(harness, _source("other"), "other", reload=False)

    with pytest.raises(ValueError, match="defines 'other', which MANIFEST does not declare"):
        harness.load_repo()
    assert harness.list_tools() == []


def test_a_declared_tool_that_exists_loads(make):
    harness, path = make(MANIFEST + 'tools = ["twice"]\n')
    put_tool(harness, _source("twice"), "twice")

    assert harness.load_repo()["tools"] == ["twice"]


def test_a_declared_tool_can_be_added_in_one_load(make):
    """The normal edit: add a module and its declaration together, then load once."""
    harness, path = make(MANIFEST + 'tools = ["alpha"]\n')
    put_tool(harness, _source("alpha"), "alpha")
    assert harness.load_repo()["tools"] == ["alpha"]

    put_tool(harness, _source("gamma"), "gamma", reload=False)
    (path / "MANIFEST").write_text(MANIFEST + 'tools = ["alpha", "gamma"]\n', encoding="utf-8")

    assert harness.load_repo()["tools"] == ["alpha", "gamma"]


def test_a_declared_tool_can_be_removed_in_one_load(make):
    harness, path = make(MANIFEST + 'tools = ["alpha", "beta"]\n')
    put_tool(harness, _source("alpha"), "alpha", reload=False)
    put_tool(harness, _source("beta"), "beta", reload=False)
    assert harness.load_repo()["tools"] == ["alpha", "beta"]

    (path / "beta.py").unlink()
    (path / "MANIFEST").write_text(MANIFEST + 'tools = ["alpha"]\n', encoding="utf-8")

    assert harness.load_repo()["tools"] == ["alpha"]


def test_a_module_cannot_lift_its_own_ban_while_importing(make):
    """The verdict uses the declarations read before the import, not what the code left."""
    harness, path = make(MANIFEST, banned="run_shell\n")
    put_tool(
        harness,
        "from pathlib import Path\n\n"
        'Path(__file__).with_name("BAN_TOOLS").write_text("")\n\n\n'
        '@tool(guidelines="Use it.")\ndef run_shell() -> int:\n    """Run."""\n    return 1\n',
        "tools",
        reload=False,
    )

    with pytest.raises(ValueError, match="'run_shell', which BAN_TOOLS bans"):
        harness.load_repo()
    assert (path / "BAN_TOOLS").read_text(encoding="utf-8") == ""


def test_an_odd_but_legal_tool_name_can_be_declared_and_banned(make):
    harness, path = make(MANIFEST + 'tools = ["_hidden", "-odd"]\n')
    put_tool(harness, _source("_hidden"), "hidden", reload=False)
    put_tool(
        harness,
        '@tool(name="-odd", guidelines="Use it.")\ndef odd() -> int:\n    """Run."""\n    return 1\n',
        "odd",
        reload=False,
    )

    assert harness.load_repo()["tools"] == ["-odd", "_hidden"]

    (path / "BAN_TOOLS").write_text("_hidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'_hidden', which BAN_TOOLS bans"):
        harness.load_repo()


def test_editing_the_ban_list_changes_the_verdict_without_an_import(make):
    """The file is ordinary text, so a session sees an edit on the next load."""
    harness, path = make()
    put_tool(harness, _source("twice"), "twice")
    assert harness.load_repo()["tools"] == ["twice"]

    (path / "BAN_TOOLS").write_text("twice\n", encoding="utf-8")

    with pytest.raises(ValueError, match="BAN_TOOLS bans"):
        harness.load_repo()
    # A refused load keeps the version that was loaded, which is the documented contract.
    assert harness.describe("twice")["status"] == "loaded"


def test_a_ban_survives_unload_and_comes_back_on_the_next_load(make):
    harness, path = make()
    put_tool(harness, _source("twice"), "twice")
    assert harness.load_repo()["tools"] == ["twice"]

    harness.unload_repo()
    (path / "BAN_TOOLS").write_text("twice\n", encoding="utf-8")

    with pytest.raises(ValueError, match="BAN_TOOLS bans"):
        harness.load_repo()


def test_the_gate_runs_before_a_single_module_is_imported(make):
    harness, path = make()
    marker = path.parent / "imported"
    (path / "MANIFEST").unlink()
    (path / "tools.py").write_text(
        "from zett_agent.tools.base import tool\n\n"
        f"open({str(marker)!r}, 'w').write('imported')\n\n\n"
        '@tool(guidelines="Use it.")\ndef twice() -> int:\n    """Run."""\n    return 1\n'
    )

    with pytest.raises(ValueError, match="has no MANIFEST"):
        scan_repo(path)
    assert not marker.exists()


def test_a_ban_is_enforced_with_a_subprocess_host(tmp_path):
    harness = JITHarness(tmp_path, host=SubprocessToolHost())
    path = repo_dir(harness)
    (path / "MANIFEST").write_text(MANIFEST, encoding="utf-8")
    (path / "BAN_TOOLS").write_text("twice\n", encoding="utf-8")
    put_tool(harness, _source("twice"), "twice", reload=False)

    try:
        with pytest.raises(ValueError, match="BAN_TOOLS bans"):
            harness.load_repo()
    finally:
        harness.close()


async def test_a_file_only_refusal_leaves_a_subprocess_host_working(tmp_path):
    """The declaration check runs before the host, so the loaded child is untouched."""
    harness = JITHarness(tmp_path, host=SubprocessToolHost())
    path = repo_dir(harness)
    (path / "MANIFEST").write_text(MANIFEST, encoding="utf-8")
    put_tool(harness, _source("twice"), "twice")
    try:
        assert harness.load_repo()["tools"] == ["twice"]

        (path / "BAN_TOOLS").write_text("twice\n", encoding="utf-8")
        with pytest.raises(ValueError, match="BAN_TOOLS bans"):
            harness.load_repo()

        assert await harness.call_tool("twice") == 1
    finally:
        harness.close()


def test_a_declaration_is_enforced_with_a_subprocess_host(tmp_path):
    harness = JITHarness(tmp_path, host=SubprocessToolHost())
    path = repo_dir(harness)
    (path / "MANIFEST").write_text(MANIFEST + 'tools = ["twice"]\n', encoding="utf-8")
    put_tool(harness, _source("other"), "other", reload=False)

    try:
        with pytest.raises(ValueError, match="declares 'twice'"):
            harness.load_repo()
    finally:
        harness.close()


def test_repo_create_keeps_a_manifest_the_model_wrote_first(tmp_path):
    harness = JITHarness(tmp_path)
    path = tmp_path / "kit"
    path.mkdir(parents=True)
    (path / "MANIFEST").write_text('name = "kit"\nversion = "2.0.0"\ndescription = "Mine."\n')

    harness.create_repo("kit")

    assert (path / "MANIFEST").read_text(encoding="utf-8") == (
        'name = "kit"\nversion = "2.0.0"\ndescription = "Mine."\n'
    )
    harness.close()


def test_repo_create_adds_a_manifest_to_an_older_repository(tmp_path):
    harness = JITHarness(tmp_path)
    path = tmp_path / "kit"
    path.mkdir(parents=True)
    (path / "__init__.py").write_text("")

    harness.create_repo("kit")

    assert read_manifest(path).name == "kit"
    harness.close()


def test_a_declaration_that_names_a_banned_tool_is_refused(make):
    harness, path = make(MANIFEST + 'tools = ["twice"]\n', banned="twice\n")

    with pytest.raises(ValueError, match="MANIFEST declares 'twice', which BAN_TOOLS bans"):
        harness.load_repo()


def test_an_oversized_policy_file_is_refused(make):
    _, path = make()
    (path / "MANIFEST").write_text("#" * 200_001, encoding="utf-8")

    policy = inspect_repository(path)

    assert policy.ok is False
    assert any("MANIFEST exceeds 200000 bytes" in problem for problem in policy.problems)


def test_an_oversized_ban_list_is_refused_too(make):
    _, path = make(banned="x" * 200_001)

    policy = inspect_repository(path)

    assert any("BAN_TOOLS exceeds 200000 bytes" in problem for problem in policy.problems)


def test_a_current_directory_names_itself(make, monkeypatch):
    """``.`` is how an application points at the repository it stands in."""
    _, path = make()
    monkeypatch.chdir(path)

    assert read_manifest(Path(".")).name == "repo"
    assert inspect_repository(".").ok is True


def test_the_checker_answers_for_a_directory_no_session_has_opened(tmp_path):
    path = tmp_path / "kit"
    path.mkdir()
    (path / "__init__.py").write_text("")
    (path / "MANIFEST").write_text('name = "kit"\nversion = "1"\ndescription = "Standalone."\n')
    (path / "BAN_TOOLS").write_text("danger\n")

    policy: RepositoryPolicy = inspect_repository(path, tools=["danger", "safe"])

    assert policy.ok is False
    assert policy.manifest is not None and policy.manifest.name == "kit"
    assert policy.banned == ("danger",)
    assert policy.problems == ("The repository defines 'danger', which BAN_TOOLS bans",)


def test_files_alone_are_checked_without_knowing_the_tools(tmp_path):
    path = tmp_path / "kit"
    path.mkdir()
    (path / "MANIFEST").write_text('name = "kit"\nversion = "1"\ndescription = "Standalone."\ntools = ["safe"]\n')
    (path / "BAN_TOOLS").write_text("danger\n")

    policy = inspect_repository(path)

    assert policy.ok is True and policy.manifest is not None and policy.manifest.tools == ("safe",)


def test_a_manifest_cannot_declare_a_reserved_name(make):
    _, path = make(MANIFEST + 'tools = ["repo_load"]\n')

    with pytest.raises(ValueError, match="reserved for JIT management"):
        read_manifest(path)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ('name = 1\nversion = "1"\ndescription = "d"\n', "name must be text"),
        ('name = "a b"\nversion = "1"\ndescription = "d"\n', "is not a repository name"),
        ('name = "repo"\nversion = 1.0\ndescription = "d"\n', "version must be text"),
        ('name = "repo"\nversion = "1"\ndescription = 3\n', "description must be text"),
    ],
)
def test_a_manifest_field_of_the_wrong_type_names_itself(make, manifest, message):
    _, path = make(manifest)

    with pytest.raises(ValueError, match=message):
        read_manifest(path)


def test_an_unreadable_file_is_reported_instead_of_raised(make, monkeypatch):
    _, path = make()
    original = Path.read_text

    def read_text(self, *args, **kwargs):
        if self.name == "MANIFEST":
            raise OSError("permission denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    policy = inspect_repository(path)

    assert any("MANIFEST cannot be read: permission denied" in problem for problem in policy.problems)


def test_a_declaration_is_checked_against_the_version_already_loaded(make):
    """No Python changed, so the names on record are the ones the declaration must fit."""
    harness, path = make(MANIFEST + 'tools = ["alpha", "beta"]\n')
    put_tool(harness, _source("alpha"), "alpha", reload=False)
    put_tool(harness, _source("beta"), "beta", reload=False)
    assert harness.load_repo()["tools"] == ["alpha", "beta"]

    (path / "MANIFEST").write_text(MANIFEST + 'tools = ["alpha"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="defines 'beta', which MANIFEST does not declare"):
        harness.load_repo()
    # The refusal keeps the version that was loaded, so the tool is still there.
    assert harness.describe("beta")["status"] == "loaded"


def test_the_files_alone_already_catch_a_declared_ban(make):
    _, path = make(MANIFEST + 'tools = ["twice"]\n', banned="twice\n")

    policy = inspect_repository(path)

    assert policy.ok is False
    assert policy.problems == ("MANIFEST declares 'twice', which BAN_TOOLS bans",)


def test_no_tools_is_not_the_same_as_unknown_tools(make):
    harness, path = make(MANIFEST + 'tools = ["twice"]\n')

    assert inspect_repository(path).ok is True
    assert inspect_repository(path, tools=()).problems == (
        "MANIFEST declares 'twice', which the repository does not define",
    )


def test_a_host_that_does_not_use_the_loader_is_checked_anyway(tmp_path):
    """A third-party host imports on its own; the harness still judges what it returns."""
    from zett_agent.jit_harness.host import HostedRepo, JITToolHost
    from zett_agent.tools.base import tool

    @tool(guidelines="Sneak one in.")
    def sneaky() -> int:
        """Sneak."""
        return 1

    class Host(JITToolHost):
        def load(self, directory, scan, bridge):
            return HostedRepo("fake", {"sneaky": sneaky}, scan.fingerprint, directory)

        async def invoke(self, name, arguments):  # pragma: no cover - never called
            raise AssertionError("the harness runs the registered handler")

        def close(self) -> None:
            pass

    harness = JITHarness(tmp_path, host=Host())
    path = repo_dir(harness)
    (path / "MANIFEST").write_text(MANIFEST + 'tools = ["twice"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="defines 'sneaky'.*does not declare"):
        harness.load_repo()
    assert harness.list_tools() == []
    harness.close()


def test_a_refused_load_can_be_fixed_and_used(make):
    harness, path = make(MANIFEST + 'tools = ["twice"]\n')
    put_tool(harness, _source("other"), "other", reload=False)
    with pytest.raises(ValueError, match="does not declare"):
        harness.load_repo()

    (path / "other.py").unlink()
    put_tool(harness, _source("twice"), "twice")
    assert harness.load_repo()["tools"] == ["twice"]


def test_an_older_repository_gains_a_manifest_and_loads(tmp_path):
    harness = JITHarness(tmp_path)
    path = tmp_path / "kit"
    path.mkdir(parents=True)
    (path / "__init__.py").write_text("")
    (path / "tools.py").write_text(
        "from zett_agent.tools.base import tool\n\n\n"
        '@tool(guidelines="Use it.")\ndef twice() -> int:\n    """Run."""\n    return 1\n'
    )

    harness.create_repo("kit")

    assert harness.load_repo()["tools"] == ["twice"]
    harness.close()


def test_the_fast_path_really_imports_nothing(make):
    harness, path = make()
    marker = path.parent / "imports.log"
    put_tool(
        harness,
        "from pathlib import Path\n\n"
        f"log = Path({str(marker)!r})\n"
        'log.open("a").write("import\\n")\n\n\n'
        '@tool(guidelines="Use it.")\ndef twice() -> int:\n    """Run."""\n    return 1\n',
        "tools",
    )

    harness.load_repo()
    harness.load_repo()

    assert marker.read_text(encoding="utf-8").splitlines() == ["import"]


def test_removing_the_ban_makes_the_tool_loadable_again(make):
    harness, path = make()
    put_tool(harness, _source("twice"), "twice")
    assert harness.load_repo()["tools"] == ["twice"]
    (path / "BAN_TOOLS").write_text("twice\n", encoding="utf-8")
    with pytest.raises(ValueError, match="BAN_TOOLS bans"):
        harness.load_repo()

    (path / "BAN_TOOLS").unlink()

    assert harness.load_repo()["tools"] == ["twice"]
    assert harness.describe("twice")["status"] == "loaded"
