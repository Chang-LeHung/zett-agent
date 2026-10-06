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
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "diagrams",
]
html_theme = "pydata_sphinx_theme"
html_title = "Zett Agent"
#: Canonical base for the GitHub Pages project site.
html_baseurl = "https://chang-lehung.github.io/zett-agent/"
html_theme_options = {
    "navigation_depth": 4,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "show_prev_next": True,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "secondary_sidebar_items": ["page-toc"],
    "github_url": "https://github.com/Chang-LeHung/zett-agent",
}
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["docs.css"]
html_sidebars = {"**": ["global-navigation.html"]}
html_context = {"default_mode": "light"}
copybutton_exclude = ".linenos, .gp, .go"
mermaid_output_format = "raw"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "both"
napoleon_custom_sections = [("Snippet", "Examples"), ("Guidelines", "Notes")]
exclude_patterns = ["_build", "_ext", "_examples", "design.md", "session-storage.md"]
