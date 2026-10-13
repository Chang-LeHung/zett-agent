"""The Chinese documentation: complete catalogs, a working build, and a language switcher."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sphinx", reason="Install the docs group or run make docs-check")
pytest.importorskip("pydata_sphinx_theme")
pytest.importorskip("babel")
from babel.messages.pofile import read_po
from bs4 import BeautifulSoup

from translation_check import check_catalog

ROOT = Path(__file__).resolve().parents[1]
CATALOGS = ROOT / "docs" / "locale" / "zh_CN" / "LC_MESSAGES"
#: API pages are generated from docstrings and stay English.
NOT_TRANSLATED = ("_generated", "_sphinx_design_static")


def copy_docs(workspace: Path) -> Path:
    """Copy the sources, never the build output, so tests do not write into the checkout."""
    source = workspace / "source"
    shutil.copytree(
        ROOT / "docs",
        source,
        ignore=shutil.ignore_patterns("_build", "_generated", "__pycache__", "*.mo"),
    )
    return source


def sphinx(source: Path, output: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sphinx", "-q", "-E", *flags, str(source), str(output)],
        capture_output=True,
        text=True,
        timeout=240,
    )


@pytest.fixture(scope="module")
def chinese(tmp_path_factory):
    """Build the Chinese site exactly as ``make docs-zh`` does, treating warnings as errors."""
    workspace = tmp_path_factory.mktemp("chinese")
    source = copy_docs(workspace)
    output = workspace / "html" / "zh"
    result = sphinx(source, output, "-W", "--keep-going", "-D", "language=zh_CN", "-b", "html")
    assert result.returncode == 0, result.stdout + result.stderr
    return output


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    """Extract the messages the current English source defines."""
    workspace = tmp_path_factory.mktemp("messages")
    source = copy_docs(workspace)
    output = workspace / "gettext"
    result = sphinx(source, output, "-b", "gettext")
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def catalogs():
    return sorted(CATALOGS.rglob("*.po"))


def test_every_catalog_is_translated_and_keeps_its_markup():
    problems = [problem for path in catalogs() for problem in check_catalog(path)]
    assert not problems, "\n".join(problems[:20])


def test_catalogs_cover_every_english_message(extracted):
    """An edited English paragraph without a matching catalog entry would render untranslated."""
    stale = []
    for template in sorted(extracted.rglob("*.pot")):
        domain = template.relative_to(extracted).with_suffix("")
        if domain.parts[0] in NOT_TRANSLATED:
            continue
        catalog = CATALOGS / domain.with_suffix(".po")
        if not catalog.exists():
            stale.append(f"{domain}: no catalog; run make docs-pot")
            continue
        current = {message.id for message in read_po(template.open("rb")) if message.id}
        translated = {message.id for message in read_po(catalog.open("rb")) if message.id}
        if current - translated:
            stale.append(f"{domain}: {len(current - translated)} message(s) are not in the catalog; run make docs-pot")
        if translated - current:
            stale.append(f"{domain}: {len(translated - current)} obsolete message(s); run make docs-pot")
    assert not stale, "\n".join(stale)


def test_the_chinese_site_declares_its_language_and_translates_the_page(chinese):
    soup = BeautifulSoup((chinese / "learn" / "jit-harness.html").read_text(), "html.parser")
    assert soup.html["lang"] == "zh-CN"
    assert soup.select_one("h1").get_text(strip=True).rstrip("#") == "构建并复用 JIT 能力"
    assert soup.select_one("nav.zett-navigation .zett-docs-home").get_text(strip=True) == "文档"
    assert 'content="能自己锻造所需工具的智能体' in (chinese / "index.html").read_text()


def test_the_chinese_sidebar_keeps_its_groups_with_translated_captions(chinese):
    soup = BeautifulSoup((chinese / "learn" / "jit-harness.html").read_text(), "html.parser")
    section = soup.select_one("nav.zett-navigation li.toctree-l1.current")
    headings = [item.get_text(strip=True) for item in section.select("li.zett-group-heading")]
    assert headings == ["基础", "对话", "能力", "生产环境"]
    assert soup.select_one("nav.zett-navigation li.toctree-l2.current > a").get_text(strip=True) == "JIT harness"


def test_the_chinese_landing_page_keeps_its_structure(chinese):
    soup = BeautifulSoup((chinese / "index.html").read_text(), "html.parser")
    assert soup.select_one(".zett-hero-title strong").get_text(strip=True) == "所需工具"
    assert len(soup.select("ul.zett-flow > li")) == 5
    primary = soup.select_one(".zett-actions p:first-child a")
    assert primary["href"] == "learn/first-agent.html"
    assert "第一个智能体" in primary.get_text()
    assert len(soup.select(".zett-path")) == 4


def test_the_switcher_leads_to_the_same_page_in_each_language(chinese):
    soup = BeautifulSoup((chinese / "learn" / "jit-harness.html").read_text(), "html.parser")
    links = {link["hreflang"]: link["href"] for link in soup.select(".zett-language-menu a")}
    # The English site is the parent of ``zh/``; from zh/learn/ that is two levels up.
    assert links == {"en": "../../learn/jit-harness.html", "zh-CN": "../../zh/learn/jit-harness.html"}
    current = soup.select_one(".zett-language-menu a.active")
    assert current["hreflang"] == "zh-CN" and current["aria-current"] == "true"
    alternates = {tag["hreflang"]: tag["href"] for tag in soup.select("link[rel=alternate][hreflang]")}
    assert alternates == links


def test_api_reference_pages_stay_english_in_the_chinese_site(chinese):
    soup = BeautifulSoup((chinese / "_generated" / "Agent.html").read_text(), "html.parser")
    assert soup.html["lang"] == "zh-CN"
    assert "Run a model, execute requested tools, and repeat until an answer is ready." in soup.get_text()
