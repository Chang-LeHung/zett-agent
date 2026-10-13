"""Check that a translation catalog is complete and keeps the source markup intact.

Run it on catalogs directly while translating::

    uv run python tests/translation_check.py docs/locale/zh_CN/LC_MESSAGES/learn/tools.po

A paragraph is translated as a unit, but the markup inside it is data: a role
that points to the wrong page, a dropped literal, or a ``**strong**`` glued to
a Chinese character (reStructuredText only recognises inline markup next to
whitespace or punctuation) would silently change the rendered page.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from babel.messages.pofile import read_po, write_po
from docutils.utils import punctuation_chars

#: CJK ideographs, kana, hangul, and the fullwidth block: characters that do not
#: count as punctuation or whitespace for reStructuredText inline markup rules.
CJK = "\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af"

LITERAL = re.compile(r"``(?P<text>.+?)``", re.S)
ROLE = re.compile(r":(?P<role>[A-Za-z][\w:+.-]*):`(?P<body>[^`]+)`")
REFERENCE = re.compile(r"(?<![\w`:])`(?P<body>[^`<]+(?:<[^`>]+>)?)`(?P<anon>__?)")
STRONG = re.compile(r"\*\*(?P<text>[^*]+)\*\*")
EMPHASIS = re.compile(r"(?<![\w*])\*(?P<text>[^*\s][^*]*)\*(?![\w*])")

#: Characters docutils allows right before inline markup starts / right after it ends.
_AFTER_START = frozenset(punctuation_chars.openers + punctuation_chars.delimiters + "'\"([{<-/:")
_BEFORE_END = frozenset(
    punctuation_chars.closers
    + punctuation_chars.delimiters
    + punctuation_chars.closing_delimiters
    + "'\")]}>-/:.,;!?\\"
)

MARKUP = re.compile(
    r"``.+?``"
    r"|:[A-Za-z][\w:+.-]*:`[^`]+`"
    r"|(?<![\w`:])`[^`]+`__?"
    r"|\*\*[^*]+\*\*"
    r"|(?<![\w*])\*[^*\s][^*]*\*(?![\w*])",
    re.S,
)


def normalise(name: str) -> str:
    """Reference names compare case-insensitively with collapsed whitespace."""
    return re.sub(r"\s+", " ", name).strip().lower()


def role_targets(text: str) -> Counter[tuple[str, str]]:
    """Return ``(role, target)`` pairs; the visible title of ``title <target>`` is free."""
    found: Counter[tuple[str, str]] = Counter()
    for match in ROLE.finditer(text):
        body = match.group("body")
        titled = re.fullmatch(r"(?s).*<(?P<target>[^<>]+)>\s*", body)
        target = titled.group("target") if titled else body
        found[(match.group("role"), target.strip())] += 1
    return found


def reference_names(text: str) -> Counter[str]:
    """Return the names that hyperlink references resolve against."""
    found: Counter[str] = Counter()
    for match in REFERENCE.finditer(text):
        body = match.group("body")
        titled = re.fullmatch(r"(?s).*<(?P<target>[^<>]+)>\s*", body)
        found[normalise(titled.group("target") if titled else body)] += 1
    return found


def literals(text: str) -> Counter[str]:
    """Return the inline code spans, which are never translated."""
    return Counter(match.group("text") for match in LITERAL.finditer(text))


def _start_ok(before: str) -> bool:
    """Whether docutils accepts inline markup that begins right after ``before``."""
    return not before or before.isspace() or before in _AFTER_START


def _end_ok(after: str) -> bool:
    """Whether docutils accepts inline markup that ends right before ``after``."""
    return not after or after.isspace() or after in _BEFORE_END


def glued_markup(text: str) -> list[str]:
    """Return inline markup that docutils would not recognise in this context.

    Inline markup must start after whitespace, an opening punctuation mark, or a
    delimiter, and end before whitespace or a closing punctuation mark. A Chinese
    character, a fullwidth opening bracket after the markup, or a fullwidth
    closing bracket before it all break that rule; a backslash-space fixes it.
    """
    problems = []
    for match in MARKUP.finditer(text):
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        if not _start_ok(before):
            problems.append(f"markup after {before!r}, add '\\ ' before: {match.group(0)[:40]!r}")
        if not _end_ok(after):
            problems.append(f"markup before {after!r}, add '\\ ' after: {match.group(0)[:40]!r}")
    return problems


def fix_glue(text: str) -> str:
    """Insert ``\\ `` wherever inline markup touches a character that would break it."""
    out, last = [], 0
    for match in MARKUP.finditer(text):
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        out.append(text[last : match.start()])
        out.append("\\ " if not _start_ok(before) else "")
        out.append(match.group(0))
        out.append("\\ " if not _end_ok(after) else "")
        last = match.end()
    out.append(text[last:])
    return "".join(out)


#: Concepts this project names itself and Chinese has no settled word for. They stay
#: English in prose: ``(pattern in the source, pattern the translation must keep)``.
KEEP_ENGLISH = (
    (r"\bharness\b", r"\bharness\b"),
    (r"\bsteer(?:ing|ed)?\b", r"\bsteer(?:ing|ed)?\b"),
    (r"\bguidelines?\b", r"\bguidelines?\b"),
    (r"\binternal continuations?\b", r"\binternal continuations?\b"),
    (r"\binspection\W{0,2} tier\b", r"\binspection\W{0,2} tier\b"),
)

#: Half-width punctuation that has a full-width form; Chinese text uses the latter.
HALFWIDTH = {",": "，", ";": "；", ":": "：", "!": "！", "?": "？"}
HALFWIDTH_NEAR_CJK = re.compile(rf"(?P<after>(?<=[{CJK}])[,;:!?])|(?P<before>[,;:!?](?=[{CJK}]))")

#: Renderings that were invented for those concepts and must not come back.
RETIRED_RENDERINGS = ("使用准则", "调用示例", "内部续接", "内部续写", "内部消息", "检查层级", "引导")

CODE = re.compile(r"``.+?``|:[A-Za-z][\w:+.-]*:`[^`]+`|`[^`]+`_{1,2}", re.S)


def prose(text: str) -> str:
    """Return the text without code, roles and links, which are checked separately."""
    return CODE.sub(" ", text)


def kept_english(source: str, translation: str) -> list[str]:
    """Return the project concepts the translation replaced with a Chinese word."""
    problems = []
    source_prose, translation_prose = prose(source), prose(translation)
    for in_source, in_translation in KEEP_ENGLISH:
        if re.search(in_source, source_prose, re.I) and not re.search(in_translation, translation_prose, re.I):
            problems.append(f"keep the project term in English (it has no settled Chinese rendering): {in_source}")
    for retired in RETIRED_RENDERINGS:
        if retired in translation_prose:
            problems.append(f"retired rendering {retired!r}; keep the English project term instead")
    return problems


def check_message(source: str, translation: str) -> list[str]:
    """Return what is wrong with one translated paragraph, empty when it is sound."""
    if not translation.strip():
        return ["not translated"]
    problems = []
    if literals(source) != literals(translation):
        problems.append(f"inline literals differ: {sorted(literals(source))} vs {sorted(literals(translation))}")
    if role_targets(source) != role_targets(translation):
        problems.append(
            f"role targets differ: {sorted(role_targets(source).elements())} vs "
            f"{sorted(role_targets(translation).elements())}"
        )
    if reference_names(source) != reference_names(translation):
        problems.append(
            "hyperlink reference names differ; write `shown text <Original English Name>`_: "
            f"{sorted(reference_names(source).elements())} vs {sorted(reference_names(translation).elements())}"
        )
    if len(STRONG.findall(source)) != len(STRONG.findall(translation)):
        problems.append("the number of **strong** spans differs")
    if source.rstrip().endswith("::") and not translation.rstrip().endswith("::"):
        problems.append("the source ends with '::' to introduce a literal block; keep it")
    problems.extend(glued_markup(translation))
    problems.extend(kept_english(source, translation))
    for match in HALFWIDTH_NEAR_CJK.finditer(prose(translation).replace("::", PROTECTED)):
        punctuation = match.group("after") or match.group("before")
        problems.append(f"write {HALFWIDTH[punctuation]!r} instead of {punctuation!r} inside Chinese text")
    return problems


def check_catalog(path: Path) -> list[str]:
    """Return one line per problem in a ``.po`` catalog."""
    with path.open("rb") as handle:
        catalog = read_po(handle)
    problems = []
    for message in catalog:
        if not message.id:
            continue
        source = message.id if isinstance(message.id, str) else message.id[0]
        translation = message.string if isinstance(message.string, str) else message.string[0]
        if "fuzzy" in message.flags:
            problems.append(f"{path}: fuzzy entry {source[:60]!r}")
            continue
        for problem in check_message(source, translation):
            line = message.lineno
            problems.append(f"{path}:{line}: {problem} [{source[:50]!r}]")
    return problems


def fix_punctuation(text: str) -> str:
    """Use the full-width form of the punctuation that touches a Chinese character."""
    out, last = [], 0
    for match in MARKUP.finditer(text):
        out.append(_widen(text[last : match.start()]))
        out.append(match.group(0))
        last = match.end()
    out.append(_widen(text[last:]))
    return "".join(out)


#: Literal-block markers and other ``::`` sequences are ASCII by definition.
PROTECTED = "\x00"


def _widen(text: str) -> str:
    """Widen the punctuation of one stretch of prose, leaving ``::`` alone."""
    guarded = text.replace("::", PROTECTED)
    widened = re.sub(rf"(?<=[{CJK}])[,;:!?]|[,;:!?](?=[{CJK}])", lambda m: HALFWIDTH[m.group(0)], guarded)
    return widened.replace(PROTECTED, "::")


def fix_catalog(path: Path) -> int:
    """Insert missing backslash-space separators in a catalog; return how many messages changed."""
    with path.open("rb") as handle:
        catalog = read_po(handle)
    changed = 0
    for message in catalog:
        if message.id and isinstance(message.string, str) and message.string:
            fixed = fix_punctuation(fix_glue(message.string))
            if fixed != message.string:
                message.string = fixed
                changed += 1
    if changed:
        with path.open("wb") as handle:
            write_po(handle, catalog, width=0)
    return changed


def main(arguments: list[str]) -> int:
    """Print every problem found in the given catalogs; exit 1 when there are any."""
    if arguments and arguments[0] == "--fix":
        for argument in arguments[1:]:
            print(f"{argument}: {fix_catalog(Path(argument))} message(s) fixed")
        arguments = arguments[1:]
    problems = [problem for argument in arguments for problem in check_catalog(Path(argument))]
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s) in {len(arguments)} catalog(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
