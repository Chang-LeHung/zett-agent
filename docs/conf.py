"""Local documentation configuration; API content is imported from source."""

import sys
from importlib.metadata import version as package_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "Zett Agent"
author = "Chang-LeHung"
release = package_version("zett-agent")
extensions = [
    "reference",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "diagrams",
]
html_theme = "pydata_sphinx_theme"
html_title = "Zett Agent"
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"
html_baseurl = "https://chang-lehung.github.io/zett-agent/"
html_theme_options = {
    "logo": {
        "image_light": "logo.svg",
        "image_dark": "logo.svg",
        "alt_text": "Zett Agent",
        "text": "Zett Agent",
    },
    "navigation_depth": 4,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "show_prev_next": True,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "secondary_sidebar_items": ["page-toc"],
    "navbar_persistent": ["search-button"],
    "footer_start": ["zett-footer"],
    "footer_end": [],
    "primary_sidebar_end": [],
    "github_url": "https://github.com/Chang-LeHung/zett-agent",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/zett-agent/",
            "icon": "fa-solid fa-box",
        },
    ],
}
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["docs.css"]
html_sidebars = {"**": ["global-navigation.html"]}
html_context = {"default_mode": "light"}
html_show_sourcelink = False
copybutton_exclude = ".linenos, .gp, .go"
mermaid_output_format = "raw"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "both"
napoleon_custom_sections = [("Snippet", "Examples"), ("Guidelines", "Notes")]
exclude_patterns = ["_build", "_ext", "_examples", "design.md", "session-storage.md"]
