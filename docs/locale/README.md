# Translating the documentation

The English `.rst` pages are the source. Each translation is a set of gettext
catalogs under `docs/locale/<language>/LC_MESSAGES/`, one `.po` file per page.
Today that is Simplified Chinese (`zh_CN`). The API reference is generated from
docstrings and stays English.

## When you change an English page

1. Edit the `.rst` page.
2. Run `make docs-pot`. It extracts the English messages and merges them into
   every catalog; messages you added or reworded show up empty or `fuzzy`.
3. Translate them, and remove the `fuzzy` flag once the translation is right.
4. Run `uv run python tests/translation_check.py docs/locale/zh_CN/LC_MESSAGES/<page>.po`.
5. Run `make docs` to build both languages with warnings as errors.

`make check` fails when a catalog is missing a message or has an obsolete one,
so a page cannot ship half translated by accident.

## Markup inside a message

A paragraph is translated as a unit, but its markup is data:

- Keep inline code (` ``like this`` `), role targets (`:doc:`, `:ref:`, `:class:`)
  and URLs exactly. Only the visible title of ``:doc:`title <path>` `` is translated.
- reStructuredText recognises inline markup only after whitespace or punctuation.
  In Chinese, put a backslash and a space where markup touches a character
  that breaks it: 使用\ ``@tool``\ 装饰器. `--fix` inserts the separators, and the
  full-width punctuation below, for you:
  `uv run python tests/translation_check.py --fix <file>.po`.
- Link to a section with a label and `:ref:`, not `` `Section title`_ ``, because
  the title itself is translated.
- A message ending in `::` introduces a code block and must keep the `::`.
- Chinese text uses full-width punctuation (`，：；！？`). The checker flags
  half-width punctuation next to a Chinese character and `--fix` converts it;
  ASCII punctuation stays inside code, roles, URLs and a `::` marker.
- Write each message on one logical line; a line break between two Chinese
  characters renders as a space.

## Terms that stay in English

Some words are this project's own concepts, and Chinese has no settled word for
them. Inventing one makes the page harder to search, harder to match with the
code, and different from page to page. Keep these in English, in prose as well
as in headings:

| Keep | Why |
| --- | --- |
| `JIT harness`, `harness` | the name of the feature and its package |
| `internal continuation` | a model step an extension queues itself; there is no common word |
| `steering` | the user injecting guidance mid-run; also the `SteeringExtension` |
| `guidelines`, `snippets` | the tool decorator's arguments and docstring sections |
| `inspection tier` | the JIT tools that read retained details |

Words with an established rendering are translated: 智能体, 工具, 钩子, 中间件,
压缩, 检查点, 会话. `translation_check.py` fails when a message drops one of the
English terms above or brings back a retired rendering, so the rule holds as the
catalogs grow. Add a new project term to `KEEP_ENGLISH` in the same commit as its
first page.

## Adding a language

Add it to `LANGUAGES` and `UI_STRINGS` in `docs/_ext/languages.py`, run
`uv run --group docs sphinx-intl update -p docs/_build/gettext -d docs/locale -l <code>`,
add a build line to `.github/workflows/docs.yml` and the `docs` target in the
`Makefile`, and translate the catalogs.
