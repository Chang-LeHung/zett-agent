"""Command-line entry point for the ZettCode TUI."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from zett_agent import ReasoningEffort

from .config import ProviderName, ZettCodeConfig, default_model, provider_api_key
from .runtime import ZettCodeRuntime
from .tui import ZettCodeTUI


def parse_args(argv: list[str] | None = None) -> ZettCodeConfig:
    """Parse CLI arguments into a validated immutable configuration."""
    default_provider = os.getenv("ZETTCODE_PROVIDER") or ProviderName.DEEPSEEK.value
    parser = argparse.ArgumentParser(description="Run the ZettCode terminal coding agent")
    parser.add_argument("workspace", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--provider", choices=[item.value for item in ProviderName], default=default_provider)
    parser.add_argument("--model")
    parser.add_argument("--api-key", help="Prefer provider environment variables to avoid shell history exposure")
    parser.add_argument("--base-url")
    parser.add_argument("--responses-api", action="store_true")
    parser.add_argument("--session")
    parser.add_argument("--database", type=Path, default=Path.home() / ".zettcode" / "sessions.sqlite3")
    parser.add_argument(
        "--reasoning-effort",
        choices=[item.value for item in ReasoningEffort],
        default=ReasoningEffort.MEDIUM.value,
    )
    parser.add_argument("--serial-tools", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=36)
    parser.add_argument("--compaction-max-tokens", type=int, default=128_000)
    parser.add_argument("--compaction-keep-tokens", type=int, default=32_000)
    args = parser.parse_args(argv)
    try:
        provider = ProviderName(args.provider)
    except ValueError:
        parser.error(f"Unsupported provider: {args.provider!r} (check ZETTCODE_PROVIDER)")
    try:
        return ZettCodeConfig(
            workspace=args.workspace,
            provider=provider,
            model=args.model or default_model(provider),
            api_key=args.api_key or provider_api_key(provider),
            database=args.database,
            session_id=args.session,
            base_url=args.base_url,
            responses_api=args.responses_api,
            reasoning_effort=ReasoningEffort(args.reasoning_effort),
            parallel_tool_call=not args.serial_tools,
            max_iterations=args.max_iterations,
            compaction_max_tokens=args.compaction_max_tokens,
            compaction_keep_tokens=args.compaction_keep_tokens,
        )
    except ValueError as error:
        parser.error(str(error))


async def async_main(config: ZettCodeConfig) -> None:
    """Own runtime lifecycle around the full-screen application."""
    runtime = await ZettCodeRuntime.create(config)
    try:
        await ZettCodeTUI(runtime).run()
    finally:
        await runtime.aclose()


def main() -> None:
    """Installed console-script entry point."""
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
