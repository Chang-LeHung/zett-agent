"""Shape both tables of contents.

Sidebar: a section's pages are grouped under small headings taken from the
``:caption:`` of each toctree in the section's index page. Sphinx keeps those
captions but the theme drops them below the top level, so they are restored
here from the environment.

Right-hand "On this page" table of contents:

API pages list ``Agent.run()``, ``Agent.stream()``, and so on under ``Agent``,
which repeats the class name on every line. Under a parent entry the prefix is
dropped, and the full text moves to a tooltip together with any other label
long enough to be clipped.
"""

from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup
from sphinx import addnodes

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


def section_groups(env: Any) -> dict[str, list[tuple[str, list[str]]]]:
    """Map each section index page to its captioned groups of document names."""
    groups: dict[str, list[tuple[str, list[str]]]] = {}
    for docname, toc in env.tocs.items():
        found = [
            (node["caption"], [entry[1] for entry in node["entries"] if entry[1] in env.all_docs])
            for node in toc.findall(addnodes.toctree)
            if node.get("caption")
        ]
        if found:
            groups[docname] = found
    return groups


def group_sidebar(html: str, groups: dict[str, list[tuple[str, list[str]]]], pagename: str, app: Any) -> str:
    """Insert caption headings before the first page of every group."""
    soup = BeautifulSoup(html, "html.parser")
    for top in soup.select("li.toctree-l1"):
        link = top.find("a", recursive=False)
        pages = top.select("li.toctree-l2")
        if link is None or not pages:
            continue
        for docname, parts in groups.items():
            if app.builder.get_relative_uri(pagename, docname) != link.get("href"):
                continue
            targets = {
                app.builder.get_relative_uri(pagename, name): caption for caption, names in parts for name in names
            }
            seen: set[str] = set()
            for page in pages:
                anchor = page.find("a", recursive=False)
                href = (anchor.get("href") or "").split("#")[0] if anchor else ""
                caption = next((c for uri, c in targets.items() if uri.split("#")[0] == href), None)
                if caption is None or caption in seen:
                    continue
                seen.add(caption)
                heading = soup.new_tag("li", attrs={"class": "zett-group-heading", "role": "presentation"})
                heading.string = caption
                page.insert_before(heading)
    return str(soup)


def regroup_sidebar(app: Any, pagename: str, templatename: str, context: dict[str, Any], doctree: Any) -> None:
    """Wrap the sidebar generator so each section shows its groups."""
    generate = context.get("generate_toctree_html")
    if generate is None:
        return
    groups = section_groups(app.env)

    def generate_grouped(kind: str, *args: Any, **kwargs: Any) -> Any:
        result = generate(kind, *args, **kwargs)
        if kind != "sidebar":
            return result
        return group_sidebar(str(result), groups, pagename, app)

    context["generate_toctree_html"] = generate_grouped


def setup(app: Any) -> dict[str, bool]:
    """Run after the theme registers its generators."""
    app.connect("html-page-context", wrap_page_toc, priority=900)
    app.connect("html-page-context", regroup_sidebar, priority=900)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
