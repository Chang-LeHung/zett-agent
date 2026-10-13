"""Shorten the right-hand "On this page" table of contents.

API pages list ``Agent.run()``, ``Agent.stream()``, and so on under ``Agent``,
which repeats the class name on every line. Under a parent entry the prefix is
dropped, and the full text moves to a tooltip together with any other label
long enough to be clipped.
"""

from __future__ import annotations

from typing import Any

#: Labels longer than this get a tooltip, because CSS may clip them.
TOOLTIP_FROM = 28


def shorten_entries(toc: Any) -> Any:
    """Rewrite one rendered TOC in place and return it."""
    if not toc or not hasattr(toc, "select"):
        return toc
    for parent in toc.select("li"):
        child_list = parent.find("ul", recursive=False)
        link = parent.find("a", recursive=False)
        if child_list is None or link is None:
            continue
        prefix = link.get_text(strip=True) + "."
        for child in child_list.select("li > a"):
            label = child.select_one("span.pre") or child
            text = label.get_text()
            if text.startswith(prefix) and len(text) > len(prefix):
                child["title"] = text
                label.string = text[len(prefix) :]
    for link in toc.select("a"):
        text = link.get_text(" ", strip=True)
        if len(text) > TOOLTIP_FROM and not link.get("title"):
            link["title"] = text
    return toc


def wrap_page_toc(app: Any, pagename: str, templatename: str, context: dict[str, Any], doctree: Any) -> None:
    """Wrap the theme's TOC generator once it has been installed."""
    generate = context.get("generate_toc_html")
    if generate is None:
        return

    def generate_short_toc(kind: str = "html") -> Any:
        return shorten_entries(generate(kind))

    context["generate_toc_html"] = generate_short_toc


def setup(app: Any) -> dict[str, bool]:
    """Run after the theme registers ``generate_toc_html``."""
    app.connect("html-page-context", wrap_page_toc, priority=900)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
