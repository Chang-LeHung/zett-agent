"""Policy validation, resolution to concrete paths, protected-path discovery, and the environment."""

from pathlib import Path

import pytest

from zett_agent.sandbox import policy as policy_module
from zett_agent.sandbox.binaries import shell_argv
from zett_agent.sandbox.errors import SandboxError, SandboxPolicyError, SandboxUnavailableError
from zett_agent.sandbox.policy import (
    DEFAULT_INHERITED_ENVIRONMENT,
    NetworkMode,
    SandboxPolicy,
    build_environment,
    command_arguments,
    resolve_policy,
)


def test_errors_form_one_hierarchy_and_a_policy_error_is_a_value_error():
    assert issubclass(SandboxPolicyError, SandboxError) and issubclass(SandboxPolicyError, ValueError)
    assert issubclass(SandboxUnavailableError, SandboxError)


def test_defaults_are_the_secure_ones():
    policy = SandboxPolicy()

    assert policy.writable_paths == () and policy.network is NetworkMode.NONE and policy.protect_defaults is True
    assert policy.inherit_environment == DEFAULT_INHERITED_ENVIRONMENT
    assert SandboxPolicy.read_only() == policy
    assert (
        "AWS_SECRET_ACCESS_KEY" not in policy.inherit_environment and "OPENAI_API_KEY" not in policy.inherit_environment
    )


def test_workspace_write_builds_a_policy_from_roots(tmp_path):
    policy = SandboxPolicy.workspace_write(
        tmp_path, str(tmp_path / "a"), network=NetworkMode.FULL, protect_defaults=False
    )

    assert policy.writable_paths == (tmp_path, tmp_path / "a")
    assert policy.network is NetworkMode.FULL and policy.protect_defaults is False


def test_paths_must_be_absolute_and_tilde_is_expanded():
    with pytest.raises(SandboxPolicyError, match="must be absolute"):
        SandboxPolicy(writable_paths=("relative/dir",))
    assert SandboxPolicy(denied_read_paths=("~/.ssh",)).denied_read_paths == (Path.home() / ".ssh",)


@pytest.mark.parametrize(
    "bad",
    [
        {"writable_paths": "/tmp"},
        {"writable_paths": (1,)},
        {"writable_paths": ("/tmp/\0x",)},
        {"network": "wifi"},
        {"protect_defaults": "yes"},
        {"inherit_environment": "PATH"},
        {"inherit_environment": ("A=B",)},
        {"inherit_environment": ("",)},
    ],
)
def test_invalid_policies_are_rejected(bad):
    with pytest.raises(SandboxPolicyError):
        SandboxPolicy(**bad)


def test_network_accepts_the_enum_value_as_a_string():
    assert SandboxPolicy(network="full").network is NetworkMode.FULL


def test_the_policy_is_immutable():
    with pytest.raises(AttributeError):
        SandboxPolicy().network = NetworkMode.FULL


def test_resolution_requires_existing_writable_paths_and_refuses_the_root(tmp_path):
    with pytest.raises(SandboxPolicyError, match="does not exist"):
        resolve_policy(SandboxPolicy.workspace_write(tmp_path / "missing"))
    with pytest.raises(SandboxPolicyError, match="filesystem root"):
        resolve_policy(SandboxPolicy.workspace_write(Path(tmp_path.anchor)))


def test_writable_paths_are_resolved_through_symlinks_and_deduplicated(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)

    resolved = resolve_policy(SandboxPolicy.workspace_write(real, tmp_path / "link", protect_defaults=False))

    assert resolved.writable == (real.resolve(),)
    assert resolved.protected_write == () and resolved.denied_read == ()


def test_explicit_denials_are_resolved_without_needing_to_exist(tmp_path):
    resolved = resolve_policy(
        SandboxPolicy(denied_write_paths=(tmp_path / "w",), denied_read_paths=(tmp_path / "r",), protect_defaults=False)
    )

    assert resolved.denied_write == (tmp_path.resolve() / "w",)
    assert resolved.denied_read == (tmp_path.resolve() / "r",)
    assert resolved.unwritable == resolved.denied_write


def test_defaults_hide_existing_credentials_and_sockets_only(tmp_path):
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".docker").mkdir()
    (tmp_path / ".docker" / "config.json").write_text("{}")

    resolved = resolve_policy(SandboxPolicy.read_only(), home=tmp_path)
    names = {
        path.relative_to(tmp_path.resolve()).as_posix()
        for path in resolved.denied_read
        if tmp_path.resolve() in path.parents
    }

    assert names == {".ssh", ".docker/config.json"}
    assert resolve_policy(SandboxPolicy(protect_defaults=False), home=tmp_path).denied_read == ()


def test_default_protection_always_lists_the_root_level_names_even_when_absent(tmp_path):
    root = tmp_path.resolve()

    resolved = resolve_policy(SandboxPolicy.workspace_write(root), home=tmp_path / "nohome")

    protected = set(resolved.protected_write)
    assert {root / ".bashrc", root / ".zshrc", root / ".gitconfig", root / ".vscode", root / ".idea"} <= protected
    assert {root / ".git" / "hooks", root / ".git" / "config"} <= protected


def test_protection_finds_nested_repositories_and_dangerous_names_to_a_bounded_depth(tmp_path):
    root = tmp_path.resolve()
    (root / "a" / "b" / ".git" / "hooks").mkdir(parents=True)
    (root / "a" / ".vscode").mkdir()
    (root / "a" / "b" / ".bashrc").write_text("x")
    (root / "a" / "b" / "c").mkdir()
    (root / "a" / "b" / "c" / ".bashrc").write_text("four levels down")
    (root / "node_modules" / "pkg" / ".git" / "hooks").mkdir(parents=True)

    protected = set(resolve_policy(SandboxPolicy.workspace_write(root), home=tmp_path / "x").protected_write)

    assert root / "a" / "b" / ".git" / "hooks" in protected and root / "a" / "b" / ".git" / "config" in protected
    assert root / "a" / ".vscode" in protected and root / "a" / "b" / ".bashrc" in protected
    assert root / "a" / "b" / "c" / ".bashrc" not in protected
    assert not any("node_modules" in path.parts for path in protected)


def test_a_git_file_in_a_worktree_is_not_treated_as_a_repository(tmp_path):
    root = tmp_path.resolve()
    (root / "wt").mkdir()
    (root / "wt" / ".git").write_text("gitdir: /elsewhere")

    protected = resolve_policy(SandboxPolicy.workspace_write(root), home=tmp_path / "x").protected_write

    assert root / "wt" / ".git" / "hooks" not in protected


def test_a_huge_tree_is_searched_in_part_and_says_so(tmp_path, monkeypatch):
    for index in range(8):
        (tmp_path / f"d{index}").mkdir()
    monkeypatch.setattr(policy_module, "_PROTECTED_SEARCH_ENTRIES", 3)

    resolved = resolve_policy(SandboxPolicy.workspace_write(tmp_path), home=tmp_path / "x")

    assert len(resolved.warnings) == 1 and "only part of the tree" in resolved.warnings[0]


def test_an_unreadable_directory_does_not_stop_the_search(tmp_path, monkeypatch):
    (tmp_path / "locked").mkdir()
    real = policy_module.os.scandir

    def scandir(path):
        if Path(path).name == "locked":
            raise PermissionError("denied")
        return real(path)

    monkeypatch.setattr(policy_module.os, "scandir", scandir)

    assert resolve_policy(SandboxPolicy.workspace_write(tmp_path), home=tmp_path / "x").protected_write


def test_a_default_read_protection_that_would_hide_a_writable_path_is_dropped_with_a_warning(tmp_path):
    ssh = tmp_path / ".ssh"
    (ssh / "project").mkdir(parents=True)

    resolved = resolve_policy(SandboxPolicy.workspace_write(ssh / "project"), home=tmp_path)

    assert ssh.resolve() not in resolved.denied_read
    assert any("default read protection" in warning and "skipped" in warning for warning in resolved.warnings)


def test_an_explicit_read_denial_that_hides_a_writable_path_is_a_mistake(tmp_path):
    (tmp_path / "work").mkdir()

    with pytest.raises(SandboxPolicyError, match="would hide the writable path"):
        resolve_policy(SandboxPolicy(writable_paths=(tmp_path / "work",), denied_read_paths=(tmp_path,)))


def test_a_denial_inside_a_writable_path_is_fine(tmp_path):
    (tmp_path / "work" / "private").mkdir(parents=True)

    resolved = resolve_policy(
        SandboxPolicy(
            writable_paths=(tmp_path / "work",),
            denied_read_paths=(tmp_path / "work" / "private",),
            protect_defaults=False,
        )
    )

    assert resolved.denied_read == ((tmp_path / "work" / "private").resolve(),)


def test_the_environment_keeps_only_the_named_variables():
    host = {"PATH": "/bin", "HOME": "/h", "OPENAI_API_KEY": "sk-secret", "AWS_SECRET_ACCESS_KEY": "x"}

    environment = build_environment(("PATH", "HOME", "MISSING"), source=host, case_insensitive=False)

    assert environment == {"PATH": "/bin", "HOME": "/h"}


def test_extra_variables_are_added_and_win():
    environment = build_environment(
        ("PATH",), {"PATH": "/mine", "TOKEN": "t"}, source={"PATH": "/bin"}, case_insensitive=False
    )

    assert environment == {"PATH": "/mine", "TOKEN": "t"}


def test_windows_matches_names_without_regard_to_case():
    host = {"Path": r"C:\bin", "SystemRoot": r"C:\Windows", "Secret": "x"}

    environment = build_environment(DEFAULT_INHERITED_ENVIRONMENT, {"PATH": "mine"}, source=host, case_insensitive=True)

    assert environment == {"SystemRoot": r"C:\Windows", "PATH": "mine"}
    assert build_environment(("PATH",), source=host, case_insensitive=False) == {}


@pytest.mark.parametrize("extra", [{"": "x"}, {"A=B": "x"}, {"A\0": "x"}, {"A": 1}, {"A": "x\0"}, {1: "x"}])
def test_invalid_extra_variables_are_rejected(extra):
    with pytest.raises(SandboxPolicyError):
        build_environment((), extra, source={}, case_insensitive=False)


def test_extra_must_be_a_mapping():
    with pytest.raises(SandboxPolicyError, match="mapping"):
        build_environment((), [("A", "b")], source={}, case_insensitive=False)


def test_the_real_host_environment_is_the_default_source(monkeypatch):
    monkeypatch.setenv("ZETT_SANDBOX_PROBE", "1")

    assert build_environment(("ZETT_SANDBOX_PROBE",)) == {"ZETT_SANDBOX_PROBE": "1"}


def test_command_arguments_wrap_a_string_in_a_shell_and_keep_a_sequence_whole():
    assert command_arguments("echo hi | cat") == (*shell_argv(), "-c", "echo hi | cat")
    assert command_arguments("ls", shell="/bin/bash") == ("/bin/bash", "-c", "ls")
    assert command_arguments(["python3", "a b.py"]) == ("python3", "a b.py")
    assert command_arguments(("ls",)) == ("ls",)


@pytest.mark.parametrize("command", ["", "   ", [], [""], [1], ["a\0b"], "a\0b"])
def test_invalid_commands_are_rejected(command):
    with pytest.raises(SandboxPolicyError):
        command_arguments(command)
