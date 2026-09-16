# ZettCode

ZettCode is a focused terminal coding agent built on `zett-agent`. Its TUI owns
raw terminal input, component layout, mouse interaction, and cell-level
differential rendering directly; it does not use prompt-toolkit or another TUI
framework.

## Run

DeepSeek:

```bash
export DEEPSEEK_API_KEY=...
uv run --directory backend/zettcode zettcode --provider deepseek --model deepseek-chat
```

OpenAI or an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
uv run --directory backend/zettcode zettcode --provider openai --model gpt-5-mini
uv run --directory backend/zettcode zettcode \
  --provider openai --model my-model --base-url https://example.com/v1
```

Pass a workspace path as the final argument. It defaults to the current directory.
Session data is stored in `~/.zettcode/sessions.sqlite3` by default.

## Keys

- `Enter`: send
- `Esc`, then `Enter`: insert a newline
- `Ctrl-C`: stop the active request, or clear the input
- `Ctrl-D`: delete the next character; exit when the input is empty and the agent is idle
- `Ctrl-A` / `Ctrl-E`, `Home` / `End`: move to the start or end of the current line
- `Ctrl-B` / `Ctrl-F`, `Left` / `Right`: move by one character
- `Alt-B` / `Alt-F`, `Ctrl-Left` / `Ctrl-Right`: move by one word
- `Ctrl-Home` / `Ctrl-End`, `Alt-<` / `Alt->`: move to the start or end of the whole input
- `Ctrl-U` / `Ctrl-K`: delete to the start or end of the current line
- `Ctrl-W` / `Alt-Backspace`, `Alt-D`: delete the previous or next word
- `Ctrl-Y`: paste the last text removed with a line or word deletion
- `Ctrl-Z` / `Ctrl-_`: undo the latest input edit
- `Up` / `Down`: move between input lines, then browse submitted input history
- `Ctrl-P` / `Ctrl-N`: browse submitted input history directly
- `Ctrl-R`: search backward through submitted input history
- `Ctrl-L`: redraw the terminal
- `Ctrl-T`: expand or collapse the latest thinking block
- `Page Up` / `Page Down`: scroll history while keeping the composer focused
- Click the transcript and use `Up` / `Down`, `Home` / `End` to browse it
- Drag in the transcript to select text; dragging past an edge automatically scrolls
- Double-click a transcript row to select the complete visible line
- `Shift`-click extends the existing selection
- Completed mouse selections are copied to the system clipboard automatically
- `Ctrl-C` also copies a transcript selection and returns focus to the input
- `Tab` / `Shift-Tab`: complete slash commands

Thinking is collapsed by default. Click a Thinking row to inspect it; moving the
pointer outside that block collapses it again. Tool calls keep their output and
completion status visible in the transcript. Active thinking and tool rows use a
low-frequency moving highlight wave with an animated activity icon. Tool results show a five-row preview by default;
click a completed tool row to expand it. Expanded output is still bounded so a
large command result cannot take over the terminal or exhaust renderer memory.

Assistant text renders common terminal-friendly Markdown: bold and emphasis,
headings, inline and fenced code, links, quotes, unordered bullet lists, and
GFM-style tables. Tables preserve left, center, and right alignment, account for
wide CJK characters, and fit or truncate columns to the current terminal width.

Use `/help` inside the application to see session commands.
