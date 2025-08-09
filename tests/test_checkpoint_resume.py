import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.training.simple import train
from src.data.loader import InstructionSample


def test_resume_continues_step(monkeypatch):
    ckpt = Path("models/training_state.pth")
    if ckpt.exists():
        ckpt.unlink()
    samples = [
        InstructionSample("i", "x", "y"),
        InstructionSample("i", "x", "y"),
    ]
    cfg = {
        "num_epochs": 1,
        "batch_size": 1,
        "num_workers": 0,
        "tokenizer_path": "models/spm_bpe_8k.model",
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    train(samples, cfg)
    first = torch.load(ckpt)
    cfg.update({"num_epochs": 2})
    train(samples, cfg, resume=True)
    second = torch.load(ckpt)
    assert second["epoch"] == first["epoch"] + 1
    assert second["global_step"] > first["global_step"]
    assert second["scheduler"]["last_epoch"] > first["scheduler"]["last_epoch"]
