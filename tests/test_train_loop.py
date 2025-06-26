import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from src.training.simple import _train_epoch


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 4):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, pad_id: int = 0
    ) -> torch.Tensor:
        batch, seq = tgt.shape
        return torch.zeros(batch, seq, self.vocab_size, requires_grad=True)


class DummyOptim:
    def zero_grad(self) -> None:
        pass

    def step(self) -> None:
        pass


def test_batch_consumption() -> None:
    dummy_loader = [
        (torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long))
        for _ in range(5)
    ]
    tokenizer = type("Tok", (), {"vocab_size": 4, "pad_id": 0})()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        torch.optim.Adam([torch.tensor(1.0, requires_grad=True)]), T_max=1
    )
    loss, _, _, _ = _train_epoch(
        dummy_loader,
        DummyModel(),
        nn.CrossEntropyLoss(),
        DummyOptim(),
        torch.cuda.amp.GradScaler(enabled=False),
        scheduler,
        tokenizer,
        "cpu",
        False,
    )
    assert loss is not None
