"""설정 입출력 테스트."""
from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import ModuleType
from typing import Iterator

import pytest

from src import config as base_config

SAMPLE_INI = """; 테스트용 기본 설정
[train]
grad_clip = 1.0             # 기울기 제한
min_lr = 0.00001
use_mixed_precision = yes
model_dim = 256
ff_dim = 1024
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
num_workers = 6
pin_memory = yes
spm_model_path = models/spm.model

[pretrain]
epochs = 20
batch_size = 48
learning_rate = 0.0002
dropout_ratio = 0.1
resume = no

[finetune]
epochs = 20
batch_size = 48
learning_rate = 0.0002
dropout_ratio = 0.1
resume = no
"""


@pytest.fixture()
def temporary_config_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ModuleType]:
    """CONFIG_PATH를 임시 경로로 지정한 모듈을 생성."""
    temporary_path = tmp_path / "trainconfig.ini"
    monkeypatch.setenv("CONFIG_PATH", str(temporary_path))
    module = importlib.reload(base_config)
    try:
        yield module
    finally:
        monkeypatch.delenv("CONFIG_PATH", raising=False)
        importlib.reload(base_config)


def test_load_config_returns_defaults_when_missing(
    temporary_config_module: ModuleType,
) -> None:
    """INI 파일이 없으면 DEFAULT_CONFIG를 반환."""
    assert (
        temporary_config_module.load_config()
        == temporary_config_module.DEFAULT_CONFIG
    )


def test_load_config_merges_ini_values(
    temporary_config_module: ModuleType,
) -> None:
    """trainconfig.ini 값을 읽어 기본 설정을 덮어쓴다."""
    temporary_config_module.CONFIG_PATH.write_text(
        SAMPLE_INI, encoding="utf-8"
    )

    loaded = temporary_config_module.load_config()

    assert loaded["grad_clip"] == 1.0
    assert loaded["min_lr"] == 1e-5
    assert loaded["use_mixed_precision"] is True
    assert loaded["num_epochs"] == 20
    assert loaded["batch_size"] == 48
    assert loaded["learning_rate"] == pytest.approx(0.0002)
    assert loaded["dropout_ratio"] == pytest.approx(0.1)


def test_save_config_updates_ini_sections(
    temporary_config_module: ModuleType,
) -> None:
    """save_config가 train/pretrain/finetune 섹션 값을 갱신."""
    temporary_config_module.CONFIG_PATH.write_text(
        SAMPLE_INI, encoding="utf-8"
    )

    payload = {
        "grad_clip": 0.5,
        "min_lr": 5e-5,
        "use_mixed_precision": False,
        "pin_memory": False,
        "num_epochs": 8,
        "batch_size": 32,
        "learning_rate": 5e-4,
        "dropout_ratio": 0.2,
        "resume": True,
    }

    temporary_config_module.save_config(payload)

    text = temporary_config_module.CONFIG_PATH.read_text(encoding="utf-8")

    train_section = re.search(r"\[train\](?P<body>.*?)(?:\n\[|\Z)", text, re.S)
    assert train_section is not None
    body = train_section.group("body")
    assert re.search(r"grad_clip\s*=\s*0.5", body)
    assert "# 기울기 제한" in body
    assert re.search(r"use_mixed_precision\s*=\s*no", body)
    assert re.search(r"pin_memory\s*=\s*no", body)

    pretrain_section = re.search(r"\[pretrain\](?P<body>.*?)(?:\n\[|\Z)", text, re.S)
    finetune_section = re.search(r"\[finetune\](?P<body>.*?)(?:\n\[|\Z)", text, re.S)
    assert pretrain_section is not None and finetune_section is not None
    for section in (pretrain_section.group("body"), finetune_section.group("body")):
        assert re.search(r"epochs\s*=\s*8", section)
        assert re.search(r"batch_size\s*=\s*32", section)
        assert re.search(r"learning_rate\s*=\s*0.0005", section)
        assert re.search(r"dropout_ratio\s*=\s*0.2", section)
        assert re.search(r"resume\s*=\s*yes", section)
