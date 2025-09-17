"""Configuration IO helper tests."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Iterator

import pytest

from src import config as base_config


@pytest.fixture()
def temporary_config_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ModuleType]:
    """Reload src.config with a temporary CONFIG_PATH and restore it after use."""
    original_env = os.environ.get("CONFIG_PATH")
    temporary_path = tmp_path / "nested" / "current.json"
    monkeypatch.setenv("CONFIG_PATH", str(temporary_path))
    module = importlib.reload(base_config)
    try:
        yield module
    finally:
        if original_env is None:
            monkeypatch.delenv("CONFIG_PATH", raising=False)
        else:
            monkeypatch.setenv("CONFIG_PATH", original_env)
        importlib.reload(base_config)


def test_load_config_returns_defaults_when_missing(
    temporary_config_module: ModuleType,
) -> None:
    """load_config should return DEFAULT_CONFIG when the file does not exist."""
    assert temporary_config_module.load_config() == temporary_config_module.DEFAULT_CONFIG


def test_load_config_handles_invalid_json(temporary_config_module: ModuleType) -> None:
    """load_config should ignore invalid JSON and fall back to defaults."""
    invalid_file = temporary_config_module.CONFIG_PATH
    invalid_file.parent.mkdir(parents=True, exist_ok=True)
    invalid_file.write_text("{ invalid", encoding="utf-8")

    loaded = temporary_config_module.load_config()

    assert loaded == temporary_config_module.DEFAULT_CONFIG


def test_save_config_persists_payload(temporary_config_module: ModuleType) -> None:
    """save_config should create parent directories and persist JSON content."""
    config_payload = temporary_config_module.DEFAULT_CONFIG.copy()
    config_payload["num_epochs"] = 5

    temporary_config_module.save_config(config_payload)

    with temporary_config_module.CONFIG_PATH.open(encoding="utf-8") as file:
        stored = json.load(file)

    assert stored["num_epochs"] == 5
    assert temporary_config_module.CONFIG_PATH.exists()
