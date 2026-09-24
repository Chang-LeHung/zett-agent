DOCS_HOST ?= 127.0.0.1
DOCS_PORT ?= 8000
UV ?= uv

.PHONY: help install lint test build check docs docs-serve docs-check docs-examples docs-ui-check

help:
	@echo "Available targets:"
	@echo "  make install    Sync the zett-agent environment from uv.lock"
	@echo "  make lint       Check formatting and linting with Ruff"
	@echo "  make test       Run the test suite with the coverage gate"
	@echo "  make build      Build the sdist and wheel into dist/"
	@echo "  make check      Run lint, tests, documentation, and ZettCode checks"
	@echo "  make docs       Build the HTML documentation"
	@echo "  make docs-serve Build and preview the documentation at http://$(DOCS_HOST):$(DOCS_PORT)"
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

docs:
	env -u VIRTUAL_ENV $(UV) run --group docs sphinx-build -E -a -W --keep-going -b html docs docs/_build/html

docs-serve: docs
	@echo "Zett Agent docs: http://$(DOCS_HOST):$(DOCS_PORT) (Ctrl+C to stop)"
	env -u VIRTUAL_ENV $(UV) run --group docs python -m http.server $(DOCS_PORT) --bind $(DOCS_HOST) --directory docs/_build/html

docs-examples:
	env -u VIRTUAL_ENV $(UV) run --group docs pytest tests/test_documentation_examples.py -q

docs-ui-check:
	env -u VIRTUAL_ENV $(UV) run --group docs-test playwright install chromium
	env -u VIRTUAL_ENV $(UV) run --group docs --group docs-test pytest tests/test_documentation.py -q

docs-check:
	env -u VIRTUAL_ENV $(UV) run --group docs ruff check docs
	env -u VIRTUAL_ENV $(UV) run --group docs pytest tests/test_documentation.py tests/test_documentation_examples.py -q
