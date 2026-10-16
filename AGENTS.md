# Development Rules

## Scope

- This repository is the standalone home of the `zett-agent` runtime
  (`src/zett_agent`). ZettCode, the terminal coding agent built on it, lives in
  its own repository.
- `zett_agent` must stay importable and installable on its own. It must never
  import the documentation extensions, a terminal client, or any host
  application.
- Keep host application vocabulary out of `zett_agent`. The runtime owns
  messages, models, providers, tools, extensions, storage contracts, and
  sessions; knowledge cards, artifacts, tags, channels, and scheduled tasks
  belong to other projects.
- Downstream packages depend on the published `zett-agent` contract only.
  Anything they need must be exported here rather than copied from here.

## Layout

- `src/zett_agent/` is the runtime; `extensions/` holds optional runtime
  extensions and `providers/` holds provider adapters.
- `src/zett_agent/_agent/` holds the private mixins that `Agent` is composed of,
  one module per concern (extension hooks, request lifecycle, the turn loop, a
  model step, tool execution, recovery). Keep every documented public method on
  `Agent` itself, because the API reference lists what `vars(Agent)` holds, and
  keep `_agent` free of imports of `zett_agent.agent` at module level.
- `tests/` covers the runtime, `examples/` holds runnable programs, and
  `docs/` is the Sphinx manual. `docs/_examples` are executed by the test
  suite, so every example must stay offline and deterministic.
- The Python package root is `src/zett_agent`; ship new modules inside it
  rather than adding new import shims.

## Tooling

- Use Python 3.10+ and `uv` for dependency management.
- Use Ruff for formatting and linting, with `line-length = 120`.
- Use mypy for type checking (`make typecheck`). It covers `src/zett_agent`,
  `examples` and `docs/_examples`, so a program the documentation shows must
  type-check too. Prefer narrowing a type to `# type: ignore`; an ignore needs a
  reason beside it, and an unused one fails the check.
- `make static` runs `static_check/`, the rules Ruff and mypy cannot express:
  a type alias is declared with `TypeAlias` or `TypeAliasType` (SC001), provider
  modules do not use `Any` (SC002), and a module does not import another
  module's underscore name (SC003). SC002 and SC003 have a baseline in
  `static_check/baseline.json` that may only shrink: a new violation fails, and
  so does an entry that is no longer needed. Fix the code, then run
  `uv run python -m static_check --update-baseline`. A line can opt out with
  `# static-check: ignore[SC00x] reason`.
- Run `make check` (Ruff, mypy, static_check, pytest with the 95% coverage gate, and the
  documentation checks) before handing work over.
- Tests must never write into the checkout or into user data. Use temporary
  directories and in-memory databases.
- Keep the documentation build warning-free: `make docs` runs Sphinx with
  `-W`, and public API pages are generated from source docstrings.

## Releases

- The version lives in `pyproject.toml`. A release is a commit that sets the
  final version, a tag named `v<version>`, and a push of that tag; the release
  workflow refuses to publish when the tag and the project version disagree.
- Publishing requires the `PYPI_API_TOKEN` repository secret. See
  `RELEASING.md`.
