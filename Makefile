DOCS_HOST ?= 127.0.0.1
DOCS_PORT ?= 8000
DOCS_OPEN ?= --open-browser
UV ?= uv
# Where `make docs` writes the site. Override it to build beside a running `make docs-live`.
DOCS_OUT ?= docs/_build/html

.PHONY: help install lint test build check docs docs-en docs-zh docs-pot docs-serve docs-live docs-check docs-examples docs-ui-check

help:
	@echo "Available targets:"
	@echo "  make install    Sync the zett-agent environment from uv.lock"
	@echo "  make lint       Check formatting and linting with Ruff"
	@echo "  make test       Run the test suite with the coverage gate"
	@echo "  make build      Build the sdist and wheel into dist/"
	@echo "  make check      Run lint, tests with coverage, and documentation checks"
	@echo "  make docs       Build the HTML documentation in English and Chinese"
	@echo "  make docs-pot   Refresh the Chinese translation catalogs from the English source"
	@echo "  make docs-serve Build and preview the documentation at http://$(DOCS_HOST):$(DOCS_PORT)"
	@echo "  make docs-live  Preview with auto-rebuild and browser reload while you edit"
	@echo "  make docs-check Validate public API coverage, examples, and local links"
	@echo "  make docs-examples  Run every offline documentation example"
	@echo "  make docs-ui-check  Verify navigation and search in Chromium"

install:
	$(UV) sync

lint:
	$(UV) run ruff format --check src tests examples
	$(UV) run ruff check src tests examples

test:
	$(UV) run pytest --cov --cov-report=term-missing

build:
	$(UV) build

check: lint test docs-check

docs: docs-en docs-zh

docs-en:
	env -u VIRTUAL_ENV $(UV) run --group docs sphinx-build -E -a -W --keep-going -b html docs $(DOCS_OUT)

# The translation is built into a subdirectory of the English site so a single
# upload serves both, and so the language switcher can link between them.
docs-zh:
	env -u VIRTUAL_ENV $(UV) run --group docs sphinx-build -E -a -W --keep-going -D language=zh_CN -b html docs $(DOCS_OUT)/zh

# Extract messages from the English source and merge them into the .po files.
# API reference pages are generated from docstrings and stay English.
docs-pot:
	rm -rf docs/_build/gettext
	env -u VIRTUAL_ENV $(UV) run --group docs sphinx-build -E -b gettext docs docs/_build/gettext
	rm -rf docs/_build/gettext/_generated docs/_build/gettext/_sphinx_design_static
	env -u VIRTUAL_ENV $(UV) run --group docs sphinx-intl update -p docs/_build/gettext -d docs/locale -l zh_CN

docs-serve: docs
	@echo "Zett Agent docs: http://$(DOCS_HOST):$(DOCS_PORT) (Ctrl+C to stop)"
	env -u VIRTUAL_ENV $(UV) run --group docs python -m http.server $(DOCS_PORT) --bind $(DOCS_HOST) --directory docs/_build/html

docs-live:
	@echo "Zett Agent docs (live reload): http://$(DOCS_HOST):$(DOCS_PORT)/ (Ctrl+C to stop)"
	env -u VIRTUAL_ENV $(UV) run --group docs sphinx-autobuild docs docs/_build/html \
		--host $(DOCS_HOST) --port $(DOCS_PORT) $(DOCS_OPEN) \
		--watch src/zett_agent --watch docs/_examples \
		--ignore 'docs/_build/*' --ignore 'docs/_generated/*' \
		-E -a -W --keep-going

docs-examples:
	env -u VIRTUAL_ENV $(UV) run --group docs pytest tests/test_documentation_examples.py -q

docs-ui-check:
	env -u VIRTUAL_ENV $(UV) run --group docs-test playwright install chromium
	env -u VIRTUAL_ENV $(UV) run --group docs --group docs-test pytest tests/test_documentation.py -q

docs-check:
	env -u VIRTUAL_ENV $(UV) run --group docs ruff check docs
	env -u VIRTUAL_ENV $(UV) run --group docs pytest tests/test_documentation.py tests/test_documentation_examples.py -q
