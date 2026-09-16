from pathlib import Path

import pytest

from zettcode.config import ProviderName, ZettCodeConfig, default_model


def test_config_resolves_paths(tmp_path: Path):
    config = ZettCodeConfig(
        workspace=tmp_path,
        provider=ProviderName.DEEPSEEK,
        model="model",
        api_key="secret",
        database=tmp_path / "data" / "sessions.sqlite3",
    )

    assert config.workspace == tmp_path.resolve()
    assert config.database == (tmp_path / "data" / "sessions.sqlite3").resolve()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"api_key": ""}, "Missing API key"),
        ({"model": ""}, "Model cannot be empty"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"compaction_keep_tokens": 0}, "compaction_keep_tokens"),
        ({"compaction_max_tokens": 10, "compaction_keep_tokens": 10}, "must be greater"),
    ],
)
def test_config_rejects_invalid_values(tmp_path: Path, changes, message):
    values = {
        "workspace": tmp_path,
        "provider": ProviderName.OPENAI,
        "model": "model",
        "api_key": "secret",
        "database": tmp_path / "sessions.sqlite3",
        **changes,
    }
    with pytest.raises(ValueError, match=message):
        ZettCodeConfig(**values)


def test_provider_defaults_are_explicit():
    assert default_model(ProviderName.DEEPSEEK) == "deepseek-chat"
    assert default_model(ProviderName.OPENAI) == "gpt-5-mini"
