from pathlib import Path

import pytest
from zett_agent import ReasoningEffort

from zettcode.cli import parse_args
from zettcode.config import ProviderName


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch):
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_API", "OPENAI_API_KEY", "ZETTCODE_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


def test_parse_args_applies_defaults(tmp_path: Path):
    config = parse_args([str(tmp_path), "--api-key", "secret"])

    assert config.workspace == tmp_path.resolve()
    assert config.provider is ProviderName.DEEPSEEK
    assert config.model == "deepseek-chat"
    assert config.reasoning_effort is ReasoningEffort.MEDIUM
    assert config.parallel_tool_call is True
    assert config.max_iterations == 36


def test_parse_args_reads_provider_and_api_key_from_the_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ZETTCODE_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")

    config = parse_args([str(tmp_path)])

    assert config.provider is ProviderName.OPENAI
    assert config.model == "gpt-5-mini"
    assert config.api_key == "from-environment"


def test_parse_args_forwards_endpoint_and_session_flags(tmp_path: Path):
    database = tmp_path / "sessions.sqlite3"
    config = parse_args(
        [
            str(tmp_path),
            "--api-key",
            "secret",
            "--session",
            "session-1",
            "--database",
            str(database),
            "--base-url",
            "https://example.test/v1",
            "--responses-api",
            "--reasoning-effort",
            "high",
            "--serial-tools",
            "--max-iterations",
            "5",
        ]
    )

    assert config.session_id == "session-1"
    assert config.database == database.resolve()
    assert config.base_url == "https://example.test/v1"
    assert config.responses_api is True
    assert config.reasoning_effort is ReasoningEffort.HIGH
    assert config.parallel_tool_call is False
    assert config.max_iterations == 5


def test_unknown_provider_override_reports_a_usage_error(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("ZETTCODE_PROVIDER", "bogus")

    with pytest.raises(SystemExit) as exit_info:
        parse_args([str(tmp_path)])

    assert exit_info.value.code == 2
    assert "Unsupported provider: 'bogus'" in capsys.readouterr().err


def test_unknown_provider_flag_reports_a_usage_error(tmp_path: Path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse_args([str(tmp_path), "--api-key", "secret", "--provider", "bogus"])

    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_missing_api_key_reports_a_usage_error(tmp_path: Path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse_args([str(tmp_path)])

    assert exit_info.value.code == 2
    assert "Missing API key for deepseek" in capsys.readouterr().err
