# Import performance work (2026-09)

Internal notes for the change that shipped as `0.1.4`. This file is deliberately
kept outside `docs/`: it records a one-off investigation, not user or developer
documentation, and it is not part of the Sphinx manual or the published package.

## Symptom

A `viztracer` profile of a downstream application showed almost all startup time
inside `import zett_agent`. A single import of any name cost roughly 1.4 s,
because `zett_agent/__init__.py` re-exported every provider adapter, built-in
extension, storage engine, and tool.

Python runs a package `__init__` before any of its submodules, so even
`import zett_agent.messages` paid for `openai`, `anthropic`, `mcp`, `sqlalchemy`,
and roughly 150 further modules.

## How it was measured

- `python -X importtime -c "import zett_agent"`, aggregated by root package, to
  attribute the cost (`importtime` reports self time and cumulative time per
  module).
- Fresh-interpreter timings: `subprocess.run([sys.executable, "-c", code])`
  repeated 9-15 times, taking the median, with the same interpreter and a warm
  page cache; a bare `pass` run measures interpreter start-up for reference.
- Before/after comparison used a `git worktree` checked out at `HEAD`
  (`721daf2`, the last commit before the work) with `PYTHONPATH` pointing at its
  `src/`, so both sides ran on the same machine and interpreter.

Baseline attribution from `-X importtime` (self time, cumulative total 1105.7 ms
for the instrumented run):

| package | self time |
| --- | ---: |
| `anthropic` | 271.7 ms |
| `openai` | 241.4 ms |
| `mcp_types` | 134.9 ms |
| `sqlalchemy` | 95.9 ms |
| `mcp` | 69.0 ms |
| `zett_agent` itself | 65.0 ms |

The runtime's own code was never the problem: it was the third-party SDKs pulled
in on every import.

## Stages

Each stage was measured after the previous one, median of fresh subprocesses.

| stage | `import zett_agent` |
| --- | ---: |
| 0. baseline (eager re-exports) | 1360.8 ms |
| 1. provider SDKs imported inside functions/classes | 278.2 ms |
| 2. packages resolved exports lazily (PEP 562) | 17.6 ms |
| 3. packages re-export nothing at all | 14.8 ms |

Interpreter start-up alone is ~14.6 ms, so the final package import is free.

### Stage 1 - import the SDK where it is used

`openai` moved into `providers/base.py` client construction and request failure
handling; `anthropic` into `providers/anthropic.py` `__init__` and `stream()`;
`mcp` into the two factory functions in `extensions/mcp.py`. Payload builders
that used the SDK's TypedDict constructors now build plain dictionaries that
still type-check against those TypedDicts under `TYPE_CHECKING`.

### Stage 2 - lazy package exports

Each package `__init__` kept its type-checking imports, declared a
name-to-module table, and resolved names through a module-level
`__getattr__`/`__dir__` per PEP 562. This made the package itself free, but every
attribute access still imported the whole graph for the module it pointed at,
and the table duplicated knowledge that already lived in the imports.

### Stage 3 - no exports at all (shipped)

The `__init__` files of `zett_agent`, `zett_agent.tools`, `zett_agent.providers`,
`zett_agent.extensions`, and `zett_agent.extensions.server_tools` now contain only
a docstring; the root package keeps `__version__`. Every name is imported from
the module that defines it:

```python
from zett_agent.agent import Agent, AgentRunConfig
from zett_agent.tools.base import tool, AgentTool
from zett_agent.messages import UserMessage
from zett_agent.extensions.coding import CodingExtension
from zett_agent.providers.openai import OpenAIProvider
```

## Final numbers

Medians for the same workloads on both sides (stage 3 versus the `HEAD`
worktree):

| case | before | after | faster |
| --- | ---: | ---: | ---: |
| `import zett_agent` | 1360.8 ms | 14.8 ms | ~92x |
| `Agent` | 1330 ms | 98 ms | 13.6x |
| `tool` (then `from zett_agent import tool`) | 1400 ms | 89 ms | 15.8x |
| `UserMessage` | 1475 ms | 46 ms | 32x |
| `CodingExtension` | 1743 ms | 181 ms | 9.7x |
| `OpenAIProvider`, import only | 1721 ms | 158 ms | 10.9x |
| agent + tool + provider + extension | 1875 ms | 233 ms | 8.0x |
| the same, then constructing `OpenAIProvider` | 1845 ms | 718 ms | 2.6x |

The remainder in the last row is the SDK itself (`import openai` ~390 ms,
`sqlalchemy` ~250 ms, `mcp` ~420 ms), paid once when that client is actually
constructed. There is nothing left to win there without dropping a dependency.

## Constraints found along the way

These are easy to trip over when optimising imports in this codebase.

1. A module-level `__getattr__` (PEP 562) only serves attribute access on the
   module object. Global lookups inside functions (`LOAD_GLOBAL`) and
   `eval`-based annotation resolution do not consult it, so `AsyncOpenAI(...)`
   in a function body raises `NameError` if the name was only provided that way.
   Deferring a provider SDK therefore needs a real import inside the function.
2. `tools/base.py` resolves the annotations of `@tool`-decorated functions with
   `typing.get_type_hints`, which evaluates them against the defining module's
   globals. A `@tool` function annotated with an OpenAI type therefore needs that
   import at module level; `extensions/tool_search.py` keeps it deliberately and
   documents why.
3. Importing a submodule binds it on its parent package. In the stage-2 design
   that binding shadowed an export with the same name (`model_request_trace`),
   because a name already present in the module `__dict__` is returned instead
   of going through `__getattr__`. Stage 3 removes the class of bug entirely.
4. `from package import name` falls back to importing a submodule of that name,
   so removing a re-export turns the call into an `ImportError` naming the
   module — which is what `tests/test_public_exports.py` asserts.

## Verification

- `make check`: Ruff, pytest with the 95% coverage gate, and the documentation
  build (`-W`) all green.
- Full suite on Python 3.10, and the project matrix on CI for 3.10-3.14.
- Live checks against the internal Responses gateway (`deepseek-flash`) for the
  Responses path, a local tool round trip, and client-side tool search; plus a
  manual Chat Completions tool round trip, because the OpenAI payload builders
  now emit dictionaries instead of TypedDict instances.
- `tests/test_public_exports.py` guards the shape of the result: the five
  packages import nothing and assign nothing (except `__version__`), expose no
  public attributes in a fresh interpreter, `from zett_agent import Agent` fails,
  importing the runtime or any adapter imports no provider SDK, and constructing
  a provider does.

## Related

- PR #9 `perf(imports): stop exporting everything from the package` (squash
  merged as `9955830`).
- Release `0.1.4`, which also flipped the PyPI classifier to
  `Development Status :: 5 - Production/Stable`.
