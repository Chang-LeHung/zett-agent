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
    "navbar_persistent": ["search-button-field"],
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
html_context = {
    "default_mode": "light",
    "social_card": f"{html_baseurl}_static/social-card.png",
    "site_description": "Agents that forge the tools they need: a small Python runtime for tool-calling agents.",
}
html_show_sourcelink = False
copybutton_exclude = ".linenos, .gp, .go"
mermaid_output_format = "raw"
# One explicit palette for both modes: light cards with dark text and amber
# borders stay legible on the light and the dark page alike.
mermaid_light_theme = "base"
mermaid_dark_theme = "base"
mermaid_init_config = {
    "startOnLoad": False,
    "themeVariables": {
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "primaryColor": "#fff7ed",
        "primaryTextColor": "#14171c",
        "primaryBorderColor": "#d97706",
        "secondaryColor": "#f1f5f9",
        "secondaryTextColor": "#14171c",
        "secondaryBorderColor": "#94a3b8",
        "tertiaryColor": "#ffffff",
        "tertiaryTextColor": "#14171c",
        "tertiaryBorderColor": "#cbd5e1",
        "mainBkg": "#fff7ed",
        "nodeBorder": "#d97706",
        "lineColor": "#8a96a6",
        "textColor": "#8a96a6",
        "edgeLabelBackground": "#f1f5f9",
        "clusterBkg": "#f8fafc",
        "clusterBorder": "#cbd5e1",
        "titleColor": "#8a96a6",
    },
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "both"
napoleon_custom_sections = [("Snippet", "Examples"), ("Guidelines", "Notes")]
exclude_patterns = ["_build", "_ext", "_examples", "design.md", "session-storage.md"]
