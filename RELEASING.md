# Releasing

`zett-agent` is published from this repository to PyPI by
`.github/workflows/release.yml`.

## One-time setup

1. Create the repository secret `PYPI_API_TOKEN` under **Settings → Secrets and
   variables → Actions** with a PyPI API token scoped to the `zett-agent`
   project. The workflow reads exactly that name.
2. Optional: add a `pypi` environment under **Settings → Environments** and
   require reviewers there if you want a manual approval before publishing.

Alternatively, drop the `password:` input from the publish step and configure
PyPI Trusted Publishing for this repository instead; the workflow already
requests `id-token: write`.

## Cutting a release

1. Set the final version in `src/zett_agent/__init__.py` (for example `0.2.0`)
   and commit it. `pyproject.toml` reads that attribute, and the release
   workflow checks the tag against it. The tagged commit must carry the
   released version, not `0.2.0.dev0`.
2. Tag and push:

   ```bash
   git tag v0.2.0
   git push origin main
   git push origin v0.2.0
   ```

3. The workflow builds the sdist and wheel, verifies the tag matches the
   project version, publishes to PyPI, and then opens a GitHub release with
   generated notes.

To rehearse the build without publishing, run `make build` and inspect
`dist/`, or start the workflow manually from the Actions tab with
`workflow_dispatch`. A manual run only builds and uploads the artifacts;
publishing and the GitHub release both require a tag push.

## CI

`.github/workflows/ci.yml` runs on every push to any branch and on every pull
request:

- Ruff format and lint checks for the runtime.
- pytest with the coverage gate on Python 3.10 through 3.14.
- Documentation build, example, and link checks.
- A `uv build` job that fails if the wheel picks up anything outside
  `zett_agent`.
