import logging
from pathlib import Path

import pytest
import torch

from src.training.simple import train
from src.data.loader import InstructionSample


def _samples() -> list[InstructionSample]:
    """Return minimal instruction samples."""
    return [InstructionSample("i", "x", "y"), InstructionSample("i", "x", "y")]


def test_mode_logging(monkeypatch, caplog) -> None:
    """Check mode log outputs for cold start and resume."""
    ckpt = Path("models/training_state.pth")
    if ckpt.exists():
        ckpt.unlink()
    cfg = {
        "num_epochs": 1,
        "batch_size": 1,
        "num_workers": 0,
        "tokenizer_path": "models/spm_bpe_8k.model",
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    caplog.set_level(logging.INFO)
    train(_samples(), cfg)
    assert "MODE=COLD_START" in caplog.text
    caplog.clear()
    cfg["num_epochs"] = 2
    train(_samples(), cfg, resume=True)
    assert "MODE=RESUME" in caplog.text


def test_no_samples(monkeypatch) -> None:
    """Raise error when dataset is empty."""
    cfg = {
        "num_epochs": 1,
        "batch_size": 1,
        "tokenizer_path": "models/spm_bpe_8k.model",
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError):
        train([], cfg)


def test_requires_epoch_or_steps(monkeypatch) -> None:
    """Ensure epochs or steps must be positive."""
    cfg = {
        "num_epochs": 0,
        "max_steps": 0,
        "batch_size": 1,
        "tokenizer_path": "models/spm_bpe_8k.model",
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError):
        train(_samples(), cfg)

