import asyncio
import base64
import importlib

import pytest
from pydantic import ValidationError

from zett_agent.tools.coding import (
    bash,
    delete_file,
    edit,
    glob,
    grep,
    read,
    read_file,
    write_file,
)


async def test_file_tools_write_read_and_replace_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    created = await write_file({"path": "notes/example.txt", "content": "one\ntwo\nthree\nfour\n"})
    assert created == "Created notes/example.txt"

    page = await read_file({"path": "notes/example.txt", "start_line": 2, "line_count": 2})
    assert page == "two\nthree\n\n... [more lines; continue with start_line=4]"

    replaced = await edit({"path": "notes/example.txt", "edits": [{"old_text": "three", "new_text": "THREE"}]})
    assert replaced == "Replaced 1 occurrence in notes/example.txt"
    assert await read_file({"path": "notes/example.txt"}) == "one\ntwo\nTHREE\nfour\n"

    overwritten = await write_file({"path": "notes/example.txt", "content": "same\nsame\n"})
    assert overwritten == "Wrote notes/example.txt"
    replaced_all = await edit(
        {"path": "notes/example.txt", "edits": [{"old_text": "same", "new_text": "changed", "replace_all": True}]}
    )
    assert replaced_all == "Replaced 2 occurrences in notes/example.txt"
    assert (tmp_path / "notes/example.txt").read_text() == "changed\nchanged\n"

    deleted = await delete_file({"path": "notes/example.txt"})
    assert deleted == "Deleted notes/example.txt"
    assert not (tmp_path / "notes/example.txt").exists()


async def test_edit_applies_a_batch_of_edits_in_one_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    await write_file({"path": "notes/batch.txt", "content": "alpha\nbeta\nbeta\n"})

    replaced = await edit(
        {
            "path": "notes/batch.txt",
            "edits": [
                {"old_text": "alpha", "new_text": "ALPHA"},
                {"old_text": "beta", "new_text": "BETA", "replace_all": True},
            ],
        }
    )

    assert replaced == "Replaced 3 occurrences in notes/batch.txt"
    assert (tmp_path / "notes/batch.txt").read_text() == "ALPHA\nBETA\nBETA\n"

    # Edits are ordered: the second one can target text the first one wrote.
    chained = await edit(
        {
            "path": "notes/batch.txt",
            "edits": [
                {"old_text": "ALPHA", "new_text": "FIRST"},
                {"old_text": "FIRST\n", "new_text": "FIRST: alpha\n"},
            ],
        }
    )

    assert chained == "Replaced 2 occurrences in notes/batch.txt"
    assert (tmp_path / "notes/batch.txt").read_text() == "FIRST: alpha\nBETA\nBETA\n"


async def test_edit_batch_failures_leave_the_file_untouched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    await write_file({"path": "notes/batch.txt", "content": "alpha\nbeta\n"})

    with pytest.raises(ValueError, match="not found"):
        await edit(
            {
                "path": "notes/batch.txt",
                "edits": [
                    {"old_text": "alpha", "new_text": "ALPHA"},
                    {"old_text": "missing", "new_text": "x"},
                ],
            }
        )

    assert (tmp_path / "notes/batch.txt").read_text() == "alpha\nbeta\n"

    # Edits are the only shape: a lone old_text or an empty batch is rejected.
    with pytest.raises(ValidationError, match="at least 1 item"):
        await edit({"path": "notes/batch.txt", "edits": []})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        await edit({"path": "notes/batch.txt", "old_text": "alpha"})
    with pytest.raises(ValidationError, match="Field required"):
        await edit({"path": "notes/batch.txt"})


async def test_file_tools_reject_invalid_or_ambiguous_operations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tool_module = importlib.import_module("zett_agent.tools.coding")
    monkeypatch.setattr(tool_module, "MAX_FILE_BYTES", 16)
    await write_file({"path": "existing.txt", "content": "same same"})
    (tmp_path / "directory").mkdir()

    with pytest.raises(ValueError, match="does not exist"):
        await read_file({"path": "missing.txt"})
    with pytest.raises(ValueError, match="does not exist"):
        await edit({"path": "missing.txt", "edits": [{"old_text": "old", "new_text": "new"}]})
    with pytest.raises(ValueError, match="does not exist"):
        await delete_file({"path": "missing.txt"})
    with pytest.raises(ValueError, match="does not exist"):
        await delete_file({"path": "directory"})
    with pytest.raises(ValueError, match="not a file"):
        await write_file({"path": "directory", "content": "text"})
    with pytest.raises(ValueError, match="already exists"):
        await write_file({"path": "existing.txt", "content": "text", "overwrite": False})
    with pytest.raises(ValueError, match="not unique"):
        await edit({"path": "existing.txt", "edits": [{"old_text": "same", "new_text": "changed"}]})
    with pytest.raises(ValueError, match="not found"):
        await edit({"path": "existing.txt", "edits": [{"old_text": "missing", "new_text": "changed"}]})
    with pytest.raises(ValueError, match="byte limit"):
        await write_file({"path": "large.txt", "content": "x" * 17})

    (tmp_path / "oversized.txt").write_text("x" * 17)
    assert await read_file({"path": "oversized.txt"}) == "x" * 17


async def test_local_tools_export_typed_schemas():
    tools = {registered.name: registered for registered in (glob, grep, read_file, write_file, edit, delete_file, bash)}
    assert set(tools) == {"glob", "grep", "read_file", "write_file", "edit", "delete_file", "bash"}
    assert tools["read_file"].parameters["properties"]["start_line"]["minimum"] == 1
    assert tools["read_file"].parameters["properties"]["line_count"]["maximum"] == 2_000
    assert tools["read_file"].parameters["properties"]["path"]["description"].startswith("Absolute path")
    assert tools["read_file"].snippet.startswith("read_file(")
    assert tools["read_file"].guidelines == (
        "Use line ranges for large files.",
        "Inspect the current content before editing a file.",
        "A trailing hint gives the next start_line and start_column when more content remains.",
    )

    with pytest.raises(ValidationError):
        await tools["read_file"]({"path": "file.txt", "start_line": 0})


async def test_glob_finds_sorted_working_directory_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "nested").mkdir(parents=True)
    (tmp_path / "src" / "z.py").write_text("")
    (tmp_path / "src" / "nested" / "a.py").write_text("")
    (tmp_path / "src" / "ignored.txt").write_text("")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("outside")
    (tmp_path / "src" / "outside.py").symlink_to(outside)

    result = await glob({"pattern": "src/**/*.py"})
    assert result == "src/nested/a.py\nsrc/outside.py\nsrc/z.py"

    limited = await glob({"pattern": "src/**/*.py", "max_results": 1})
    assert limited == "src/nested/a.py\n... [truncated; narrow the pattern or increase max_results]"

    absolute = await glob({"pattern": str(tmp_path / "src" / "**" / "*.py")})
    assert absolute == "\n".join(
        [
            str(tmp_path / "src" / "nested" / "a.py"),
            str(tmp_path / "src" / "outside.py"),
            str(tmp_path / "src" / "z.py"),
        ]
    )
    outside.unlink()


async def test_grep_searches_text_files_and_reports_locations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "first.py").write_text("class Agent:\n    pass\n")
    (tmp_path / "src" / "second.py").write_text("agent = 'lowercase'\nclass Other:\n")
    (tmp_path / "src" / "binary.py").write_bytes(b"\xff\xfe")

    result = await grep({"pattern": "agent", "file_pattern": "src/**/*.py", "case_sensitive": False, "max_results": 10})
    assert result == "src/first.py:1: class Agent:\nsrc/second.py:1: agent = 'lowercase'"

    limited = await grep({"pattern": "class|agent", "file_pattern": "src/**/*.py", "max_results": 1})
    assert limited == "src/first.py:1: class Agent:\n... [truncated; narrow the search or increase max_results]"

    with pytest.raises(ValueError, match="Invalid regular expression"):
        await grep({"pattern": "[", "file_pattern": "src/**/*.py"})
    absolute = await grep({"pattern": "Agent", "file_pattern": str(tmp_path / "src" / "*.py")})
    assert absolute == f"{tmp_path / 'src' / 'first.py'}:1: class Agent:"

    tool_module = importlib.import_module("zett_agent.tools.coding")
    monkeypatch.setattr(tool_module, "MAX_FILE_BYTES", 4)
    oversized = await grep({"pattern": "Agent", "file_pattern": "src/first.py"})
    assert oversized == ""


async def test_file_tools_accept_absolute_and_parent_paths(tmp_path, monkeypatch):
    working_directory = tmp_path / "workspace"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    absolute = tmp_path / "absolute.txt"

    created = await write_file({"path": str(absolute), "content": "before"})
    assert created == f"Created {absolute}"
    assert await read_file({"path": str(absolute)}) == "before"

    replaced = await edit({"path": "../absolute.txt", "edits": [{"old_text": "before", "new_text": "after"}]})
    assert replaced == "Replaced 1 occurrence in ../absolute.txt"
    assert await read_file({"path": "../absolute.txt"}) == "after"

    globbed = await glob({"pattern": "../*.txt"})
    assert globbed == "../absolute.txt"
    searched = await grep({"pattern": "after", "file_pattern": "../*.txt"})
    assert searched == "../absolute.txt:1: after"

    deleted = await delete_file({"path": str(absolute)})
    assert deleted == f"Deleted {absolute}"
    assert not absolute.exists()


async def test_shell_tool_reports_failure_and_retains_truncated_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_module = importlib.import_module("zett_agent.tools.output")
    monkeypatch.setattr(output_module, "MAX_OUTPUT_BYTES", 100)
    with pytest.raises(RuntimeError, match="exited with code 3") as captured:
        await bash({"command": "printf START; printf '%0200d' 0; printf END; printf 'error' >&2; exit 3"})

    message = str(captured.value)
    assert message.startswith("Command exited with code 3")
    assert "START" in message
    assert "END" in message
    assert "[stderr]\nerror" in message
    assert ".zett-tool-output" in message
    output_files = list((tmp_path / ".zett-tool-output").rglob("stdout.txt"))
    assert len(output_files) == 1
    assert output_files[0].read_text().endswith("END")

    with pytest.raises(ValueError, match="blank"):
        await bash({"command": "   "})


async def test_shell_tool_terminates_after_timeout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="timed out after 1 second"):
        await bash({"command": "while :; do :; done", "timeout_seconds": 1})


async def test_shell_tool_terminates_when_its_task_is_cancelled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task = asyncio.create_task(bash({"command": "while :; do :; done"}))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_edit_failures_tell_the_model_how_to_recover(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "code.py").write_text("def run():\n    return 1\n\nvalue = 1\nvalue = 1\n")

    # Indentation drift is the usual reason an exact match fails.
    with pytest.raises(ValueError, match="differs only in whitespace or indentation") as raised:
        await edit({"path": "code.py", "edits": [{"old_text": "def run():\n  return 1", "new_text": "x"}]})
    assert "Nothing was written" in str(raised.value)
    with pytest.raises(ValueError, match="copy old_text from its current content"):
        await edit({"path": "code.py", "edits": [{"old_text": "no such line", "new_text": "x"}]})

    # In a batch the failing edit is identified, and the earlier ones are not written.
    with pytest.raises(ValueError, match=r"Edit 2 of 2: Text to replace is not unique.*surrounding lines"):
        await edit(
            {
                "path": "code.py",
                "edits": [{"old_text": "return 1", "new_text": "return 2"}, {"old_text": "value = 1", "new_text": "x"}],
            }
        )
    assert (tmp_path / "code.py").read_text().count("return 1") == 1


def test_file_tool_guidance_steers_models_away_from_shell_edits():
    from zett_agent.tools.coding import bash, edit

    guidance = " ".join(edit.guidelines)
    assert "Always use this tool when you need to modify an existing file, instead of sed" in guidance
    assert "Avoid sed, cat, and similar shell commands" in guidance and "unless it is necessary" in guidance
    assert "bulk or mechanical rewrites" in guidance  # shell edits stay possible when needed
    assert "rather than switching to a shell command" in guidance
    assert "never" not in guidance.lower()
    assert edit.description.startswith("Edit an existing UTF-8 file")
    assert "always use this tool to modify a file" in edit.description
    # The reason is for the maintainers, not the model.
    assert "diff" not in (guidance + edit.description).lower()
    assert "Prefer read_file, grep, glob, write_file, and edit" in " ".join(bash.guidelines)


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


async def test_read_returns_an_image_with_a_description(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shot.png").write_bytes(PNG)
    # The extension is irrelevant: the signature decides.
    (tmp_path / "shot.dat").write_bytes(PNG)

    for name in ("shot.png", "shot.dat"):
        text, image = await read({"path": name})

        assert text.text == f"Image {name}: image/png, {len(PNG)} bytes"
        assert image.source.data == PNG and image.source.media_type == "image/png"
        assert image.alt_text == name


@pytest.mark.parametrize(
    ("header", "media_type"),
    [
        (b"\xff\xd8\xff\xe0" + b"\x00" * 16, "image/jpeg"),
        (b"GIF89a" + b"\x00" * 16, "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
    ],
)
async def test_read_recognizes_every_supported_image_format(tmp_path, monkeypatch, header, media_type):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "picture").write_bytes(header)

    _, image = await read({"path": "picture"})

    assert image.source.media_type == media_type


async def test_read_still_reads_text_that_merely_looks_binary_at_the_start(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "plain.txt").write_text("GIF is a format\nsecond line\n")

    assert await read({"path": "plain.txt"}) == "GIF is a format\nsecond line\n"
    (tmp_path / "empty.txt").write_bytes(b"")
    assert await read({"path": "empty.txt"}) == ""
    # `read` returns the same text as `read_file` for a text file.
    assert await read({"path": "plain.txt"}) == await read_file({"path": "plain.txt"})


async def test_read_rejects_oversized_images_and_unreadable_binary_files(tmp_path, monkeypatch):
    from zett_agent.tools import coding

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(coding, "MAX_IMAGE_BYTES", 10)
    (tmp_path / "big.png").write_bytes(PNG)
    with pytest.raises(ValueError, match="Image exceeds the 10-byte limit"):
        await read({"path": "big.png"})

    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00\x80")
    with pytest.raises(ValueError, match="not UTF-8 text"):
        await read({"path": "blob.bin"})


def test_bash_defaults_to_sixty_seconds_and_keeps_its_ceiling():
    from zett_agent.tools.coding import DEFAULT_BASH_TIMEOUT_SECONDS

    schema = bash.parameters["properties"]["timeout_seconds"]
    assert DEFAULT_BASH_TIMEOUT_SECONDS == 60
    assert (schema["default"], schema["maximum"], schema["minimum"]) == (60, 300, 1)
    assert "timeout_seconds=60" in bash.snippet


async def test_bash_without_a_timeout_runs_under_the_sixty_second_budget(tmp_path, monkeypatch):
    from zett_agent.tools import coding

    monkeypatch.chdir(tmp_path)
    budgets = []
    real_timeout = coding.timeout

    def recording_timeout(seconds):
        budgets.append(seconds)
        return real_timeout(seconds)

    monkeypatch.setattr(coding, "timeout", recording_timeout)

    assert await bash({"command": "printf ok"}) == "ok"
    assert budgets == [60]


def test_the_file_tool_is_called_edit_and_the_old_name_is_gone():
    from zett_agent.tools import coding

    assert edit.name == "edit"
    assert not hasattr(coding, "replace_in_file")
    assert "replace_in_file" not in " ".join(
        text
        for registered in (glob, grep, read_file, write_file, edit, delete_file, bash)
        for text in registered.guidelines
    )


async def test_read_file_stays_text_only_and_rejects_images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shot.png").write_bytes(PNG)

    with pytest.raises(ValueError, match="not UTF-8 text"):
        await read_file({"path": "shot.png"})
