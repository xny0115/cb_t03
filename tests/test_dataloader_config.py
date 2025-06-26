import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training import simple
import pytest
from src.data.loader import InstructionSample


@pytest.mark.gpu
def test_dataloader_uses_cfg(monkeypatch):
    captured = {}
    orig_loader = simple.DataLoader

    def spy_loader(dataset, batch_size, shuffle, collate_fn, num_workers, pin_memory, **kwargs):
        captured.update({'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': pin_memory})
        return orig_loader(dataset, batch_size=1, shuffle=shuffle, collate_fn=collate_fn, num_workers=0, pin_memory=False)

    monkeypatch.setattr(simple, 'DataLoader', spy_loader)
    simple.train([InstructionSample('i', 'x', 'y')], {'num_epochs': 1, 'batch_size': 7, 'num_workers': 3, 'pin_memory': False})

    assert captured == {'batch_size': 7, 'num_workers': 3, 'pin_memory': False}
