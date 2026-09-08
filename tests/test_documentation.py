"""Build the complete public reference in isolation and inspect rendered HTML."""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sphinx", reason="Install the docs group or run make docs-check")
pytest.importorskip("pydata_sphinx_theme")
from bs4 import BeautifulSoup

import zett_agent


@pytest.fixture(scope="module")
def documentation(tmp_path_factory):
    """Never write test output into the checked-out documentation or user data."""
    root = Path(__file__).resolve().parents[1]
    workspace = tmp_path_factory.mktemp("documentation")
    source = workspace / "source"
    output = workspace / "html"
    shutil.copytree(root / "docs", source, ignore=shutil.ignore_patterns("_build", "_generated", "__pycache__"))
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-q", "-E", "-W", "--keep-going", "-b", "html", str(source), str(output)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return source, output


def test_all_public_exports_have_reference_pages(documentation):
    source, output = documentation
    names = zett_agent.__all__
    assert len(names) == len(set(names))
    assert len(names) == len({name.casefold() for name in names})
    for name in names:
        assert (source / "_generated" / f"{name}.rst").is_file(), name
        page = output / "_generated" / f"{name}.html"
        assert page.is_file(), name
        soup = BeautifulSoup(page.read_text(), "html.parser")
        assert soup.find("h1").get_text(strip=True).startswith(name)


def test_examples_parameters_and_inherited_callbacks_are_rendered(documentation):
    _, output = documentation
    factory = BeautifulSoup((output / "_generated/create_agent.html").read_text(), "html.parser")
    assert "Maximum model calls" in factory.get_text()
    assert any("await create_agent" in block.get_text() for block in factory.select("div.highlight pre"))
    hooks = BeautifulSoup((output / "_generated/AgentExtension.html").read_text(), "html.parser")
    assert "on_state" in hooks.get_text()
    assert "before_tool_events" in hooks.get_text()
    assert any("NEW REQUEST" in block.get_text() for block in hooks.select("pre"))
    tool = BeautifulSoup((output / "_generated/write_file.html").read_text(), "html.parser")
    assert "Parameters" in tool.get_text() and "Guidelines" in tool.get_text()
    assert "content" in tool.get_text()


@pytest.mark.parametrize(
    ("name", "text", "admonition"),
    [
        ("Agent", "UserMessage", "warning"),
        ("AgentModel", "EchoModel", "note"),
        ("ModelEvent", "one terminal ModelResponse", "warning"),
        ("AgentEvent", "TOOL_COMPLETED/FAILED", "note"),
        ("SQLiteSessionStorage", "SessionView", "warning"),
        ("OpenAIProvider", "OPENAI_API_KEY", "note"),
        ("CodingExtension", "capability bundle", "warning"),
    ],
)
def test_core_docstrings_render_examples_diagrams_and_admonitions(documentation, name, text, admonition):
    """Guard rich source docstrings instead of accepting one-line API pages."""
    _, output = documentation
    soup = BeautifulSoup((output / f"_generated/{name}.html").read_text(), "html.parser")
    assert text in soup.get_text()
    assert soup.select_one(f".admonition.{admonition}") is not None
    assert soup.select("div.highlight pre")


@pytest.mark.parametrize("name", ["Agent", "AgentState", "AgentExtension", "ModelEvent", "SQLiteSessionStorage"])
def test_mermaid_diagrams_are_embedded(documentation, name):
    """Require semantic diagrams and the browser renderer on generated API pages."""
    _, output = documentation
    soup = BeautifulSoup((output / f"_generated/{name}.html").read_text(), "html.parser")
    diagram = soup.select_one("pre.mermaid")
    assert diagram is not None
    assert any(
        "mermaid" in script.get("src", "") or "import mermaid" in script.get_text() for script in soup.select("script")
    )


def test_runtime_source_keeps_ascii_diagrams():
    """Keep Mermaid syntax out of runtime docstrings while rendering it in HTML."""
    root = Path(__file__).resolve().parents[1] / "src" / "zett_agent"
    sources = [path.read_text() for path in root.rglob("*.py")]
    assert not any(".. mermaid::" in source for source in sources)
    assert sum(source.count(".. zett-diagram::") for source in sources) == 7


def test_state_diagram_return_path_stays_in_one_column():
    """A return arrow must join the same column as every vertical segment."""
    import inspect

    lines = inspect.getdoc(zett_agent.AgentState).splitlines()
    start = next(index for index, line in enumerate(lines) if "| READY " in line)
    end = next(index for index, line in enumerate(lines) if "| GENERATING " in line)
    column = lines[start].rindex("+")
    assert lines[end].rindex("+") == column
    for line in lines[start + 1 : end]:
        assert len(line) > column and line[column] == "|", line


def test_local_html_links_resolve(documentation):
    from urllib.parse import unquote, urlsplit

    _, output = documentation
    pages = {page.resolve(): BeautifulSoup(page.read_text(), "html.parser") for page in output.rglob("*.html")}
    anchors = {page: {node["id"] for node in soup.select("[id]")} for page, soup in pages.items()}
    for page in output.rglob("*.html"):
        soup = pages[page.resolve()]
        for link in soup.select("a[href]"):
            url = urlsplit(link["href"])
            if url.scheme or url.netloc:
                continue
            target = (page.parent / unquote(url.path)).resolve() if url.path else page.resolve()
            assert target.exists(), (page, link["href"])
            if url.fragment and target in pages:
                assert unquote(url.fragment) in anchors[target], (page, link["href"])


def test_getting_started_code_is_valid_python(documentation):
    _, output = documentation
    soup = BeautifulSoup((output / "learn/first-agent.html").read_text(), "html.parser")
    blocks = soup.select("div.highlight-python pre")
    assert blocks
    for block in blocks:
        for number in block.select(".linenos"):
            number.decompose()
        # AST parsing supports top-level await snippets intended for async callers.
        ast.parse(block.get_text())


@pytest.mark.parametrize("page", ["index.html", "extending/first-extension.html", "_generated/Agent.html"])
def test_global_navigation_is_available_on_every_page(documentation, page):
    _, output = documentation
    soup = BeautifulSoup((output / page).read_text(), "html.parser")
    navigation = soup.select_one("nav.zett-navigation")
    assert navigation is not None
    from urllib.parse import urlsplit

    links = [
        (output / page).parent.joinpath(urlsplit(link["href"]).path).resolve()
        if urlsplit(link["href"]).path
        else (output / page).resolve()
        for link in navigation.select("a[href]")
    ]
    for target in ("learn/index.html", "extending/first-extension.html", "examples/index.html", "reference/index.html"):
        assert (output / target).resolve() in links, (page, target)


def test_protected_extension_interfaces_have_anchors(documentation):
    _, output = documentation
    soup = BeautifulSoup((output / "_generated/ExternalEventExtension.html").read_text(), "html.parser")
    for method in ("_wait_for_external_event", "_take_external_event", "accept", "on_error"):
        assert soup.find(id=f"zett_agent.ExternalEventExtension.{method}"), method


def test_desktop_mobile_navigation_and_search(documentation):
    """Exercise actual rendered layout, navigation, search, and mobile overflow."""
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    playwright = pytest.importorskip("playwright.sync_api", reason="Run make docs-ui-check for browser verification")
    _, output = documentation
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(output)))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            base = f"http://127.0.0.1:{server.server_port}"
            page.goto(base, wait_until="networkidle")
            navigation = page.locator("nav.zett-navigation")
            assert navigation.is_visible()
            assert len(page.locator(".zett-path").all()) == 4
            page.screenshot(path=str(output / "desktop.png"), full_page=True)
            navigation.locator('a[href="extending/first-extension.html"]').click()
            assert page.locator("h1").inner_text().startswith("Write your first extension")
            assert page.locator(".bd-sidebar-secondary").is_visible()
            page.locator("button.copybtn").first.wait_for()
            page.context.grant_permissions(["clipboard-read", "clipboard-write"])
            page.locator("button.copybtn").first.click()
            page.wait_for_function("navigator.clipboard.readText().then(text => text.includes('async def on_tool'))")
            page.screenshot(path=str(output / "extension.png"), full_page=True)
            page.goto(f"{base}/_generated/Agent.html", wait_until="networkidle")
            diagram = page.locator("pre.mermaid svg").first
            diagram.wait_for(timeout=20_000)
            assert diagram.get_attribute("aria-roledescription") == "flowchart-v2"
            code_font = page.locator("div.highlight pre").first.evaluate(
                "element => getComputedStyle(element).fontFamily"
            )
            assert code_font.startswith('SFMono-Regular, "SF Mono"')
            page.goto(f"{base}/search.html?q=ExternalEventExtension", wait_until="networkidle")
            page.locator("#search-results li").first.wait_for()
            assert "ExternalEventExtension" in page.locator("#search-results").inner_text()
            page.set_viewport_size({"width": 390, "height": 844})
            page.goto(base, wait_until="networkidle")
            page.get_by_role("button", name="Site navigation", exact=True).click()
            assert navigation.is_visible()
            navigation.locator('a[href="extending/first-extension.html"]').click()
            page.locator("h1").wait_for()
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
            page.screenshot(path=str(output / "mobile.png"))
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
