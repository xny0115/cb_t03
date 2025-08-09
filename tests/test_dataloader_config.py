import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.simple import _create_loader
from src.training.helpers import PairDataset
import torch


def test_dataloader_uses_cfg(monkeypatch):
    dataset = PairDataset([([1], [1])])
    with monkeypatch.context() as m:
        m.setattr(torch.cuda, "is_available", lambda: True)
        loader = _create_loader(dataset, {"batch_size": 7, "num_workers": 3})
    assert loader.batch_size == 7
    assert loader.num_workers == 3
    assert loader.pin_memory is True
