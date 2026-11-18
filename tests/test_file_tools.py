"""The file tools: what each one does, pinned where the tools live.

One reader for text and pictures, no ``delete_file``; a text-only reader for a model that cannot take a picture;
search; and writes that are whole or not at all.
"""

from __future__ import annotations

import base64
import os
import re
import stat
import struct
import sys
import zlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from zett_agent.messages import ImageContent, TextContent
from zett_agent.tools.files import edit, glob, grep, read, read_file, write_file
from zett_agent.tools.output import MAX_OUTPUT_BYTES
from zett_agent.tools.workspace import FRAGMENT_CHARACTERS, MAX_FILE_BYTES

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


def _png(path: Path) -> Path:
    """The smallest valid PNG: one white pixel."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    path.write_bytes(data + chunk(b"IEND", b""))
    return path


@pytest.fixture
def here(tmp_path: Path, monkeypatch) -> Path:
    """A working directory with a few files in it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("class Agent:\n    pass\n")
    (tmp_path / "src" / "util.py").write_text("def agent():\n    return 1\n")
    return tmp_path


# -- read and read_file ------------------------------------------------------------------------------------------


async def test_read_returns_text_as_a_range_and_a_picture_as_an_image(here: Path):
    picture = _png(here / "dot.png")

    text = await read({"file_path": "a.txt", "offset": 2, "limit": 1})
    image = await read({"file_path": str(picture)})

    assert text.startswith("2\ttwo\n") and "one" not in text.split("...")[0]
    assert isinstance(image, list) and any(isinstance(part, ImageContent) for part in image)
    assert any(isinstance(part, TextContent) and "image/png" in part.text for part in image)


async def test_a_picture_is_recognised_by_its_bytes_and_not_by_its_name(here: Path):
    _png(here / "notes.txt")

    image = await read({"file_path": "notes.txt"})

    assert isinstance(image, list) and image[1].source.media_type == "image/png"


@pytest.mark.parametrize(
    ("header", "media_type"),
    [
        (b"\xff\xd8\xff\xe0" + b"\x00" * 16, "image/jpeg"),
        (b"GIF89a" + b"\x00" * 16, "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
    ],
)
async def test_every_supported_picture_format_is_recognised(here: Path, header: bytes, media_type: str):
    (here / "picture").write_bytes(header)

    image = await read({"file_path": "picture"})

    assert isinstance(image, list) and image[1].source.media_type == media_type


async def test_text_that_merely_starts_like_a_picture_is_still_text(here: Path):
    (here / "plain.txt").write_text("GIF is a format\nsecond line\n")
    (here / "empty.txt").write_bytes(b"")

    assert await read({"file_path": "plain.txt"}) == "1\tGIF is a format\n2\tsecond line\n"
    assert await read({"file_path": "empty.txt"}) == ""
    assert await read({"file_path": "plain.txt"}) == await read_file({"file_path": "plain.txt"})


async def test_the_text_reader_a_text_only_model_gets_never_returns_a_picture(here: Path):
    picture = _png(here / "dot.png")

    assert "one" in await read_file({"file_path": "a.txt"})
    with pytest.raises(ValueError, match="not UTF-8 text"):
        await read_file({"file_path": str(picture)})


async def test_a_range_says_how_to_go_on_when_there_is_more(here: Path):
    (here / "long.txt").write_text("".join(f"line {n}\n" for n in range(1, 11)))

    text = await read_file({"file_path": "long.txt", "offset": 3, "limit": 2})

    assert text == "3\tline 3\n4\tline 4\n\n... [more lines; continue with offset=5]"


async def test_an_offset_past_the_end_of_the_file_returns_nothing(here: Path):
    assert await read_file({"file_path": "a.txt", "offset": 50}) == ""


async def test_a_call_that_names_no_limit_gets_a_page_of_two_thousand_lines(here: Path):
    (here / "page.txt").write_text("".join(f"line {n}\n" for n in range(1, 2_006)))

    for reader in (read, read_file):
        text = await reader({"file_path": "page.txt"})

        assert text.startswith("1\tline 1\n")
        assert "2000\tline 2000" in text
        assert "2001\tline 2001" not in text
        assert text.rstrip().endswith("... [more lines; continue with offset=2001]")


async def test_a_default_page_is_also_held_to_the_byte_budget(here: Path):
    (here / "wide.txt").write_text("".join("x" * 39 + "\n" for _ in range(2_000)))

    text = await read_file({"file_path": "wide.txt"})

    assert "output truncated; continue with offset=" in text
    assert "line truncated" not in text and "read it with the shell" not in text
    assert text.count("\n") < 2_000, "the byte budget, not the line default, is what cuts this page"
    assert len(text.encode("utf-8")) <= MAX_OUTPUT_BYTES + 200


@pytest.mark.parametrize("size", [FRAGMENT_CHARACTERS, FRAGMENT_CHARACTERS * 2, FRAGMENT_CHARACTERS * 2 + 10])
async def test_a_line_read_in_fragments_carries_its_number_once(here: Path, size: int):
    body = "x" * size
    (here / "long-line.txt").write_text(body + "\nsecond\n")

    text = await read_file({"file_path": "long-line.txt", "limit": 2})

    assert text.count("1\t") == 1, "a prefix per fragment would repeat the number inside the content"
    assert text.split("\t", 1)[1].split("\n", 1)[0] == body


@pytest.mark.parametrize("first", [15, 16, 17])
async def test_a_budget_that_runs_out_inside_a_line_number_emits_no_bare_number(here: Path, monkeypatch, first: int):
    from zett_agent.tools import workspace as workspace_module

    monkeypatch.setattr(workspace_module, "MAX_OUTPUT_BYTES", 20)
    (here / "budget.txt").write_text("x" * first + "\n" + "y" * 10 + "\n")

    text = await read_file({"file_path": "budget.txt", "limit": 2})

    assert "\n2" not in text.split("...")[0], "a cut inside the prefix must not look like line 2's content"
    assert "output truncated; continue with offset=2" in text


async def test_a_line_too_long_to_return_is_cut_and_the_reader_is_sent_to_the_shell(here: Path):
    (here / "wide.txt").write_text("x" * (MAX_OUTPUT_BYTES + 100) + "\nnext\n")

    text = await read_file({"file_path": "wide.txt"})

    assert "line truncated" in text and "read it with the shell" in text
    assert len(text.encode("utf-8")) <= MAX_OUTPUT_BYTES + 200


async def test_a_long_line_after_a_short_one_is_already_known_to_be_too_long(here: Path):
    """The cut is measured, not assumed: the first answer says the shell is the way, and costs no round trip."""
    (here / "late.txt").write_text("short first line\n" + "x" * (MAX_OUTPUT_BYTES * 4) + "\n")

    text = await read_file({"file_path": "late.txt"})

    assert "line truncated; line 2 is longer than" in text and "read it with the shell" in text
    assert "offset=" not in text, "there is no offset that would fit this line"


async def test_an_ordinary_line_cut_by_a_full_budget_is_reached_again_with_its_offset(here: Path):
    """A thirty-kilobyte line fits a call; the one before it used the budget up, so the next call starts there."""
    (here / "mixed.txt").write_text("".join("a" * 39 + "\n" for _ in range(1000)) + "z" * 30_000 + "\n")

    text = await read_file({"file_path": "mixed.txt", "limit": 2000})
    start = int(text.rsplit("offset=", 1)[1].rstrip("]"))
    rest = await read_file({"file_path": "mixed.txt", "offset": start, "limit": 2000})

    assert "output truncated; continue with offset=1001" in text
    assert rest.startswith("1001\tzzz") and "output truncated" not in rest, "one more call carries the whole line"


async def test_a_budget_used_up_by_earlier_lines_says_where_to_go_on_not_that_a_line_is_too_long(here: Path):
    """The line a full budget cuts is an ordinary line: `offset` is the way to the rest of the file."""
    (here / "many.txt").write_text("".join("y" * 38 + "\n" for _ in range(2000)))

    text = await read_file({"file_path": "many.txt", "limit": 2000})

    assert "output truncated; continue with offset=" in text
    assert "longer than" not in text and "read it with the shell" not in text
    start = int(text.rsplit("offset=", 1)[1].rstrip("]"))
    assert start > 1 and text.count("\n") < 2000, "several lines came back before the cut"
    assert len(text.encode("utf-8")) <= MAX_OUTPUT_BYTES + 200


async def test_a_missing_file_and_a_binary_file_are_refused_with_the_path_as_given(here: Path):
    (here / "blob.bin").write_bytes(b"\xff\xfe\x00\x81" * 50)

    with pytest.raises(ValueError, match="File does not exist: nope.txt"):
        await read({"file_path": "nope.txt"})
    with pytest.raises(ValueError, match="File is not UTF-8 text: blob.bin"):
        await read({"file_path": "blob.bin"})


async def test_an_absolute_path_is_taken_as_it_is_and_so_is_one_outside_the_directory(here: Path, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "far.txt"
    elsewhere.write_text("faraway\n")

    assert "faraway" in await read({"file_path": str(elsewhere)})


async def test_a_reader_rejects_an_image_over_the_byte_limit(here: Path, monkeypatch):
    from zett_agent.tools import files as files_module

    monkeypatch.setattr(files_module, "MAX_IMAGE_BYTES", 10)
    (here / "big.png").write_bytes(PNG)

    with pytest.raises(ValueError, match="Image exceeds the 10-byte limit: big.png"):
        await read({"file_path": "big.png"})


# -- glob and grep -----------------------------------------------------------------------------------------------


async def test_glob_lists_sorted_paths_in_the_style_of_the_pattern(here: Path):
    assert await glob({"pattern": "src/*.py"}) == "src/app.py\nsrc/util.py"
    absolute = await glob({"pattern": str(here / "src" / "*.py")})
    assert absolute.splitlines() == [str(here / "src" / "app.py"), str(here / "src" / "util.py")]
    assert await glob({"pattern": "nothing/*"}) == ""


async def test_glob_follows_a_symlink_out_of_the_directory(here: Path):
    outside = here.parent / f"{here.name}-outside.py"
    outside.write_text("outside")
    (here / "src" / "outside.py").symlink_to(outside)

    assert await glob({"pattern": "src/*.py"}) == "src/app.py\nsrc/outside.py\nsrc/util.py"
    outside.unlink()


async def test_glob_stops_at_the_limit_and_says_so(here: Path):
    result = await glob({"pattern": "src/*.py", "max_results": 1})

    assert result == "src/app.py\n... [truncated; narrow the pattern or increase max_results]"


async def test_grep_reports_path_line_and_text_and_skips_what_is_not_text(here: Path):
    (here / "src" / "blob.py").write_bytes(b"\xff\xfeAgent\x00")

    result = await grep({"pattern": "Agent", "file_pattern": "src/*.py"})

    assert result == "src/app.py:1: class Agent:"


async def test_grep_can_ignore_case_and_says_when_the_expression_is_wrong(here: Path):
    both = await grep({"pattern": "agent", "file_pattern": "src/*.py", "case_sensitive": False})

    assert both.splitlines() == ["src/app.py:1: class Agent:", "src/util.py:1: def agent():"]
    with pytest.raises(ValueError, match="Invalid regular expression"):
        await grep({"pattern": "(", "file_pattern": "src/*.py"})


async def test_grep_stops_at_the_limit_and_says_so(here: Path):
    result = await grep({"pattern": "a", "file_pattern": "src/*.py", "max_results": 2})

    assert len(result.splitlines()) == 3 and result.endswith("narrow the search or increase max_results]")


async def test_grep_cuts_a_very_long_matching_line_around_the_match(here: Path):
    (here / "wide.txt").write_text("a" * 5000 + "NEEDLE" + "b" * 5000 + "\n")

    result = await grep({"pattern": "NEEDLE", "file_pattern": "wide.txt"})

    assert "NEEDLE" in result and len(result.encode()) < 2100


async def test_a_file_over_the_size_limit_is_left_out_of_a_search(here: Path, monkeypatch):
    from zett_agent.tools import files as files_module

    monkeypatch.setattr(files_module, "MAX_FILE_BYTES", 100)
    (here / "src" / "big.py").write_text("Agent and more: " + "x" * 200)

    assert await grep({"pattern": "Agent", "file_pattern": "src/*.py"}) == "src/app.py:1: class Agent:"


async def test_a_search_is_held_to_the_byte_budget(here: Path, monkeypatch):
    from zett_agent.tools import files as files_module

    (here / "many.txt").write_text("needle\n" * 40)

    monkeypatch.setattr(files_module, "MAX_OUTPUT_BYTES", 40)
    result = await grep({"pattern": "needle", "file_pattern": "many.txt"})

    assert result.endswith("narrow the search or increase max_results]")
    assert len(result.encode("utf-8")) < 40 + 60


# -- write_file and edit -----------------------------------------------------------------------------------------


async def test_write_file_creates_then_replaces_and_says_which(here: Path):
    assert await write_file({"file_path": "new/dir/n.txt", "content": "hi\n"}) == "Created new/dir/n.txt"
    assert (here / "new" / "dir" / "n.txt").read_text() == "hi\n"

    assert await write_file({"file_path": "new/dir/n.txt", "content": "bye\n"}) == "Wrote new/dir/n.txt"
    assert (here / "new" / "dir" / "n.txt").read_text() == "bye\n"


async def test_write_file_refuses_a_directory(here: Path):
    with pytest.raises(ValueError, match="Path is not a file: src"):
        await write_file({"file_path": "src", "content": "x"})
    assert (here / "a.txt").read_text() == "one\ntwo\nthree\n"


async def test_a_write_leaves_no_stray_file_and_refuses_what_is_too_big(here: Path):
    with pytest.raises(ValueError, match="byte limit"):
        await write_file({"file_path": "big.txt", "content": "x" * (MAX_FILE_BYTES + 1)})

    assert not (here / "big.txt").exists()
    assert sorted(path.name for path in here.iterdir()) == ["a.txt", "src"]


async def test_an_absolute_path_is_written_through_and_named_back(here: Path):
    elsewhere = here.parent / "outside-write.txt"
    assert await write_file({"file_path": str(elsewhere), "content": "far\n"}) == f"Created {elsewhere}"

    replaced = await edit({"file_path": "../outside-write.txt", "old_string": "far", "new_string": "near"})
    assert replaced == "Replaced 1 occurrence in ../outside-write.txt"
    assert elsewhere.read_text() == "near\n"
    elsewhere.unlink()


async def test_edit_replaces_exact_text_once_and_says_how_many(here: Path):
    result = await edit({"file_path": "a.txt", "old_string": "two", "new_string": "2"})

    assert result == "Replaced 1 occurrence in a.txt" and (here / "a.txt").read_text() == "one\n2\nthree\n"


async def test_one_edit_changes_one_place_so_the_next_change_is_another_call(here: Path):
    first = await edit({"file_path": "a.txt", "old_string": "one", "new_string": "1"})
    second = await edit({"file_path": "a.txt", "old_string": "two", "new_string": "2"})

    assert (first, second) == ("Replaced 1 occurrence in a.txt", "Replaced 1 occurrence in a.txt")
    assert (here / "a.txt").read_text() == "1\n2\nthree\n"


async def test_edit_replace_all_counts_every_match(here: Path):
    (here / "r.txt").write_text("x x x\n")

    result = await edit({"file_path": "r.txt", "old_string": "x", "new_string": "y", "replace_all": True})

    assert result == "Replaced 3 occurrences in r.txt" and (here / "r.txt").read_text() == "y y y\n"


async def test_an_edit_with_an_empty_new_string_deletes_the_match(here: Path):
    assert (
        await edit({"file_path": "a.txt", "old_string": "two\n", "new_string": ""}) == "Replaced 1 occurrence in a.txt"
    )
    assert (here / "a.txt").read_text() == "one\nthree\n"


async def test_edit_says_what_is_wrong_with_old_string(here: Path):
    (here / "dup.txt").write_text("a\na\n")
    (here / "ind.txt").write_text("if x:\n    return 1\n")

    with pytest.raises(ValueError, match=r"not unique; found 2 matches"):
        await edit({"file_path": "dup.txt", "old_string": "a", "new_string": "b"})
    with pytest.raises(ValueError, match="differs only in whitespace or indentation"):
        await edit({"file_path": "ind.txt", "old_string": "if x:\n  return 1", "new_string": "x"})
    with pytest.raises(ValueError, match="copy old_string from its current content"):
        await edit({"file_path": "ind.txt", "old_string": "no such line", "new_string": "x"})
    with pytest.raises(ValueError, match="File does not exist: ghost.txt"):
        await edit({"file_path": "ghost.txt", "old_string": "a", "new_string": "b"})
    assert (here / "dup.txt").read_text() == "a\na\n", "a failed edit writes nothing"


async def test_an_edit_of_numbered_lines_says_the_numbers_are_not_in_the_file(here: Path):
    with pytest.raises(ValueError, match="number and a tab") as raised:
        await edit({"file_path": "a.txt", "old_string": "1\tone\n2\ttwo", "new_string": "x"})

    assert "Nothing was written" in str(raised.value)
    assert (here / "a.txt").read_text() == "one\ntwo\nthree\n", "a failed edit writes nothing"


async def test_edit_refuses_an_edit_with_nothing_to_find_and_a_call_with_a_field_it_does_not_know(here: Path):
    with pytest.raises(ValidationError):
        await edit({"file_path": "a.txt", "old_string": "", "new_string": "x"})
    with pytest.raises(ValidationError):
        await edit({"file_path": "a.txt", "old_string": "one", "new_string": "1", "replace": True})
    with pytest.raises(ValidationError):
        await edit({"file_path": "a.txt", "new_string": "1"})


async def test_a_call_with_an_argument_the_tool_does_not_have_is_refused(here: Path):
    with pytest.raises(ValidationError):
        await read({"file_path": "a.txt", "mode": "rb"})
    with pytest.raises(ValidationError):
        await read({"file_path": "a.txt", "limit": 5000})
    with pytest.raises(ValidationError):
        await read({"file_path": "a.txt", "offset": 0})


async def test_edit_refuses_a_file_over_the_read_limit_and_names_it(here: Path):
    (here / "huge.txt").write_text("x" * (MAX_FILE_BYTES + 1))

    with pytest.raises(ValueError, match=rf"File exceeds the {MAX_FILE_BYTES}-byte limit: .*huge\.txt"):
        await edit({"file_path": "huge.txt", "old_string": "x", "new_string": "y"})
    assert (here / "huge.txt").stat().st_size == MAX_FILE_BYTES + 1, "a refused edit writes nothing"


async def test_a_temporary_name_that_is_taken_makes_the_write_try_another(here: Path, monkeypatch):
    from zett_agent.tools import workspace as workspace_module

    (here / "aaaa.tmp").write_text("someone else's")
    names = iter(["aaaa", "bbbb"])
    monkeypatch.setattr(workspace_module.secrets, "token_hex", lambda size: next(names))

    assert await write_file({"file_path": "new.txt", "content": "mine\n"}) == "Created new.txt"

    assert (here / "new.txt").read_text() == "mine\n"
    assert (here / "aaaa.tmp").read_text() == "someone else's", "a name that is taken is not overwritten"
    assert not (here / "bbbb.tmp").exists(), "the temporary file is renamed into place"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not keep POSIX permission bits")
async def test_a_secret_is_never_readable_while_it_is_being_rewritten(here: Path):
    secret = here / ".env"
    secret.write_text("TOKEN=old\n")
    secret.chmod(0o600)

    opened: list[tuple[int, int]] = []
    real_open = os.open

    def recording_open(path, flags, mode=0o777):
        """Record what the temporary asked for and what the kernel gave it, which is the mode it keeps."""
        descriptor = real_open(path, flags, mode)
        if Path(path).parent == here:
            opened.append((mode, stat.S_IMODE(os.fstat(descriptor).st_mode)))
        return descriptor

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "open", recording_open)
        await edit({"file_path": ".env", "old_string": "old", "new_string": "new"})

    assert opened == [(0o600, 0o600)], "the temporary is no more readable than the file it replaces"
    assert secret.read_text() == "TOKEN=new\n"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not keep POSIX permission bits")
async def test_a_new_file_gets_the_mode_a_plain_create_would(here: Path):
    umask = os.umask(0)
    os.umask(umask)

    await write_file({"file_path": "new.txt", "content": "x"})

    assert stat.S_IMODE((here / "new.txt").stat().st_mode) == 0o666 & ~umask


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not keep POSIX permission bits")
async def test_a_write_keeps_the_permission_bits_of_the_file_it_replaces(here: Path):
    script = here / "run.sh"
    script.write_text("#!/bin/sh\necho old\n")
    script.chmod(0o755)

    await edit({"file_path": "run.sh", "old_string": "old", "new_string": "new"})

    assert script.read_text() == "#!/bin/sh\necho new\n"
    assert stat.S_IMODE(script.stat().st_mode) == 0o755, "a replaced file keeps being executable"


# -- the tools as the model sees them ----------------------------------------------------------------------------


def test_the_file_tools_carry_their_schema_and_guidance():
    tools = {registered.name: registered for registered in (glob, grep, read, read_file, write_file, edit)}
    assert set(tools) == {"glob", "grep", "read", "read_file", "write_file", "edit"}

    page = tools["read_file"]
    assert page.parameters["properties"]["offset"]["minimum"] == 1
    assert page.parameters["properties"]["limit"]["maximum"] == 2_000
    assert page.parameters["properties"]["limit"]["default"] == 2_000
    assert page.parameters["properties"]["file_path"]["description"].startswith("Absolute path")
    assert page.parameters["required"] == ["file_path"]
    assert page.snippet.startswith("read_file(")
    assert page.guidelines == (
        "Use line ranges for large files.",
        "Inspect the current content before editing a file.",
        "The line numbers are for reference; do not copy them into an edit or a write.",
    )
    assert "always use this tool to modify a file" in tools["edit"].description
    assert "The text to put in its place" in tools["edit"].parameters["properties"]["new_string"]["description"]


def test_every_tool_a_bundle_guideline_names_is_registered_in_that_bundle():
    """A guideline that names a tool the caller was not given sends the model after nothing.

    Two limits, both deliberate. Only names that cannot be read as ordinary English are checked: ``read``,
    ``edit`` and ``write`` are words too. And only the three bundles are checked, because a single-tool
    extension cannot know what else an application registers — ``[ReadExtension(), ViewImageExtension()]`` is
    handed a ``view_image`` that mentions ``read_file``, and no wording is right for both cases.
    """
    from zett_agent.extensions.coding import CodingExtension
    from zett_agent.extensions.file_system import FileSystemExtension
    from zett_agent.tools.images import view_image
    from zett_agent.tools.shell import bash

    checkable = {tool.name for tool in (read_file, glob, grep, write_file, view_image, bash)}
    for bundle in (FileSystemExtension(read_only=True), FileSystemExtension(), CodingExtension()):
        registered = {tool.name for tool in bundle.tools}
        for tool in bundle.tools:
            named = {name for name in checkable if re.search(rf"\b{name}\b", " ".join(tool.guidelines))}
            assert named <= registered, (type(bundle).__name__, tool.name, named - registered)


def test_there_is_no_delete_tool():
    from zett_agent.tools import files

    assert not hasattr(files, "delete_file")
    assert files.edit.name == "edit"
