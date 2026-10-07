"""Build the complete public reference in isolation and inspect rendered HTML."""

import ast
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from xml.etree import ElementTree

import pytest

pytest.importorskip("sphinx", reason="Install the docs group or run make docs-check")
pytest.importorskip("pydata_sphinx_theme")
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def reference_exports():
    """Return the documented names by asking the reference generator itself."""
    spec = importlib.util.spec_from_file_location("reference_ext", ROOT / "docs" / "_ext" / "reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference_exports()


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
    names = list(reference_exports())
    assert len(names) > 200
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
    assert "context.emit" in hooks.get_text()
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

    from zett_agent.agent import AgentState

    lines = inspect.getdoc(AgentState).splitlines()
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


def manual_python_blocks(output):
    """Read public manual snippets, excluding generated API/source listings."""
    pages = [output / "index.html"]
    for section in ("learn", "concepts", "extending", "examples"):
        pages.extend(sorted((output / section).glob("*.html")))
    for page in pages:
        soup = BeautifulSoup(page.read_text(), "html.parser")
        for block in soup.select("div.highlight-python pre"):
            for number in block.select(".linenos"):
                number.decompose()
            yield page, dedent(block.get_text())


def test_all_user_guide_python_snippets_parse(documentation):
    _, output = documentation
    for page, code in manual_python_blocks(output):
        try:
            ast.parse(code)
        except SyntaxError as error:
            pytest.fail(f"{page.relative_to(output)}: {error}")


def test_user_facing_tool_snippets_register_without_running_handlers(documentation):
    """Catch invalid decorator guidance that syntax-only checks cannot detect."""
    from typing import Literal

    from zett_agent.tools.base import AgentTool, ToolExecutionMode, tool

    _, output = documentation
    snippets = list(manual_python_blocks(output))
    snippets.extend(
        (ROOT / "README.md", code)
        for code in re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.S)
    )
    registered = 0
    for page, code in snippets:
        for node in ast.walk(ast.parse(code)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(
                isinstance(decorator, ast.Name)
                and decorator.id == "tool"
                or isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "tool"
                for decorator in node.decorator_list
            ):
                continue
            namespace = {"tool": tool, "ToolExecutionMode": ToolExecutionMode, "Literal": Literal}
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(page), "exec"), namespace)
            except Exception as error:
                pytest.fail(f"{page.name}: {node.name}: {error}")
            assert isinstance(namespace[node.name], AgentTool)
            registered += 1
    assert registered >= 10


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        ("configuration", "Metadata is not a prompt"),
        ("application", "Cancellation is not rollback"),
        ("project-instructions", "include_parents=False"),
        ("images", "ImageBytesSource"),
        ("mcp", "catalog__search"),
        ("testing", "httpx.MockTransport"),
    ],
)
def test_task_guides_are_navigable_and_explain_usage(documentation, page, expected):
    _, output = documentation
    soup = BeautifulSoup((output / "learn" / f"{page}.html").read_text(), "html.parser")
    assert expected in soup.get_text()
    assert soup.select("div.highlight-python pre")
    landing = BeautifulSoup((output / "index.html").read_text(), "html.parser")
    assert landing.select_one(f'nav.zett-navigation a[href="learn/{page}.html"]') is not None


def test_landing_page_exposes_installation_and_learning_paths(documentation):
    _, output = documentation
    soup = BeautifulSoup((output / "index.html").read_text(), "html.parser")
    assert soup.select_one('meta[name="description"]')["content"]
    assert soup.select_one(".zett-hero-example") is not None
    assert soup.select_one(".zett-code-note a[href='learn/first-agent.html']") is not None
    assert len(soup.select(".zett-path")) == 4
    assert "pip install zett-agent" in soup.get_text()
    assert "uv add zett-agent" in soup.get_text()
    assert soup.select_one(".zett-flow[aria-label]") is not None
    assert soup.select_one(".zett-footer") is not None
    for block in soup.select("div.highlight-python pre"):
        ast.parse(block.get_text())


def test_logo_and_favicon_keep_the_same_accessible_mark(documentation):
    _, output = documentation
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    marks = []
    for filename in ("logo.svg", "favicon.svg"):
        root = ElementTree.parse(output / "_static" / filename).getroot()
        assert root.attrib["viewBox"] == "0 0 64 64"
        assert root.find("svg:title", namespace).text == "Zett Agent"
        marks.append([path.attrib["d"] for path in root.findall("svg:path", namespace)])
    assert marks[0] and marks[0] == marks[1]


def test_creature_forms_share_a_core_but_have_distinct_silhouettes(documentation):
    _, output = documentation
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    filenames = ("logo.svg", "creature-curious.svg", "creature-focused.svg", "creature-expansive.svg")
    silhouettes = []
    eyes = []
    pupils = []
    for filename in filenames:
        root = ElementTree.parse(output / "_static" / filename).getroot()
        assert root.attrib["viewBox"] == "0 0 64 64"
        assert root.find("svg:title", namespace).text.startswith("Zett Agent")
        assert len(root.findall("svg:path", namespace)) == 1
        assert root.find("svg:path", namespace).attrib["fill"] == "url(#iris)"
        assert root.find("svg:g[@data-part='eyes']", namespace).attrib["fill"] == "#fff"
        assert root.find("svg:g[@data-part='pupils']", namespace).attrib["fill"] == "#2c224b"
        silhouettes.append(root.find("svg:path", namespace).attrib["d"])
        eyes.append([eye.attrib for eye in root.findall("svg:g[@data-part='eyes']/svg:ellipse", namespace)])
        pupils.append([pupil.attrib for pupil in root.findall("svg:g[@data-part='pupils']/svg:ellipse", namespace)])
    assert len(set(silhouettes)) == len(filenames)
    assert len(eyes[0]) == 2 and all(pair == eyes[0] for pair in eyes)
    assert len(pupils[0]) == 2 and all(pair == pupils[0] for pair in pupils)
    for eye, pupil in zip(eyes[0], pupils[0], strict=True):
        assert float(pupil["cx"]) > float(eye["cx"])
        assert float(pupil["cy"]) < float(eye["cy"])
    monochrome = ElementTree.parse(output / "_static" / "mark.svg").getroot()
    assert monochrome.find("svg:path", namespace).attrib["d"] == silhouettes[0]
    assert monochrome.find("svg:path", namespace).attrib["fill"] == "currentColor"
    preview = ElementTree.parse(output / "_static" / "brand-preview.svg").getroot()
    assert preview.find("svg:defs/svg:path[@id='body']", namespace).attrib["d"] == silhouettes[0]
    for name, silhouette in zip(("curious", "focused", "expansive"), silhouettes[1:], strict=True):
        paths = preview.findall(f"svg:defs/svg:g[@id='{name}']/svg:path", namespace)
        assert len(paths) == 1 and paths[0].attrib["d"] == silhouette


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        ("examples/sessions.html", "Raw roles: system, user, assistant, system, user, assistant"),
        ("examples/compaction.html", "Raw messages: 6; checkpoint boundary: 3; raw tail: 3"),
        ("examples/storage-adapter.html", "load, append:system, append:user, append:assistant"),
        ("learn/on-demand.html", "Skill loaded: code-review"),
        ("learn/project-instructions.html", "Project guidance: concise reviews"),
        ("learn/images.html", "Received text and image in order."),
        ("learn/mcp.html", "MCP connection closed."),
        ("learn/application.html", "Timeout cleaned up; session reused."),
        ("learn/testing.html", "Mock provider reply: Hello from a mock endpoint."),
    ],
)
def test_documented_output_matches_the_downloadable_example(documentation, page, expected):
    _, output = documentation
    soup = BeautifulSoup((output / page).read_text(), "html.parser")
    assert any(expected in block.get_text() for block in soup.select("div.highlight-text pre"))


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
        assert soup.find(id=f"zett_agent.extensions.external.ExternalEventExtension.{method}"), method


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
            assert page.locator(".navbar-brand .logo__image.only-light").is_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
            page.locator(".zett-install label").filter(has_text="uv").click()
            assert page.get_by_text("uv add zett-agent", exact=True).is_visible()
            page.locator(".zett-install label").filter(has_text="pip").click()
            page.keyboard.press("ControlOrMeta+k")
            assert page.locator("#pst-search-dialog").is_visible()
            page.keyboard.press("Escape")
            page.get_by_role("button", name="Color mode", exact=True).click()
            page.locator('.theme-change-button[data-mode="dark"]').filter(visible=True).click()
            page.wait_for_function("document.documentElement.dataset.theme === 'dark'")
            assert page.locator(".navbar-brand .logo__image.only-dark").is_visible()
            primary = page.locator(".zett-actions .sd-btn-primary")
            playwright.expect(primary).to_have_css("background-color", "rgb(193, 180, 255)")
            playwright.expect(primary).to_have_css("color", "rgb(34, 27, 53)")
            hero_icon = page.locator(".zett-hero-identity img")
            playwright.expect(hero_icon).to_have_css("filter", "none")
            playwright.expect(hero_icon).to_have_css("background-color", "rgba(0, 0, 0, 0)")
            page.screenshot(path=str(output / "dark.png"), full_page=True)
            page.get_by_role("button", name="Color mode", exact=True).click()
            page.locator('.theme-change-button[data-mode="light"]').filter(visible=True).click()
            page.wait_for_function("document.documentElement.dataset.theme === 'light'")
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
            page.goto(f"{base}/learn/providers.html", wait_until="networkidle")
            labels = page.locator("label.sd-tab-label")
            playwright.expect(labels.filter(has_text="Anthropic")).to_have_css("color", "rgb(104, 98, 115)")
            for name in ("Anthropic", "Google", "DeepSeek", "Ollama", "OpenAI"):
                labels.filter(has_text=name).click()
            assert "OpenAIProvider" in page.locator(".sd-tab-content").filter(visible=True).inner_text()
            page.set_viewport_size({"width": 390, "height": 844})
            playwright.expect(page.locator(".bd-sidebar-secondary")).to_be_hidden()
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
            page.screenshot(path=str(output / "mobile-provider.png"))
            page.goto(base, wait_until="networkidle")
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
            page.screenshot(path=str(output / "mobile-home.png"), full_page=True)
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
