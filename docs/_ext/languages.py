"""Link every page to its twin in the other language.

The English site is built at the site root and each translation into a
subdirectory named after its language (``/zh/``). Every page is reachable in
both, so the switcher keeps the reader on the same page. Links are relative,
which makes them work for a local preview as well as for the deployed site.
"""

from __future__ import annotations

from typing import Any

#: Site languages: Sphinx code, URL directory ("" for the root), menu label.
LANGUAGES = (
    ("en", "", "English"),
    ("zh_CN", "zh", "简体中文"),
)
HTML_LANGUAGE_TAGS = {"en": "en", "zh_CN": "zh-CN"}

#: Strings that live in templates rather than in a page, so no ``.po`` file holds them.
UI_STRINGS = {
    "en": {
        "documentation": "Documentation",
        "tagline": "Agents that forge the tools they need.",
        "quickstart": "Quickstart",
        "jit_harness": "JIT harness",
        "api_reference": "API reference",
        "license": "MIT license",
        "description": "Agents that forge the tools they need: a small Python runtime for tool-calling agents.",
    },
    "zh_CN": {
        "documentation": "文档",
        "tagline": "能自己锻造所需工具的智能体。",
        "quickstart": "快速开始",
        "jit_harness": "JIT harness",
        "api_reference": "API 参考",
        "license": "MIT 许可证",
        "description": "能自己锻造所需工具的智能体：一个用于工具调用智能体的小型 Python 运行时。",
    },
}


def add_language_context(app: Any, pagename: str, templatename: str, context: dict[str, Any], doctree: Any) -> None:
    """Expose ``zett_languages`` and ``zett_alternates`` to the templates."""
    current = app.config.language or "en"
    directories = {code: directory for code, directory, _ in LANGUAGES}
    depth = pagename.count("/")
    # Climb to this language's root, then out of its directory if it has one.
    up = "../" * depth
    to_site_root = up + ("../" if directories.get(current) else "")
    target = pagename + ".html" if pagename != "index" else "index.html"
    links = []
    for code, directory, label in LANGUAGES:
        prefix = f"{directory}/" if directory else ""
        links.append(
            {
                "code": code,
                "html_lang": HTML_LANGUAGE_TAGS[code],
                "label": label,
                "url": to_site_root + prefix + target,
                "current": code == current,
            }
        )
    context["zett_languages"] = links
    context["zett_ui"] = UI_STRINGS.get(current, UI_STRINGS["en"])
    context["zett_language_label"] = next(link["label"] for link in links if link["current"])


def setup(app: Any) -> dict[str, bool]:
    """Register the context hook; it needs no build state, so builds stay parallel-safe."""
    app.connect("html-page-context", add_language_context)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
