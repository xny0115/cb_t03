"""Utility training helpers."""
from __future__ import annotations

from typing import Any, Iterable
import torch
from torch import optim
from torch.utils.data import DataLoader, IterableDataset
import logging
import time


class EarlyStopper:
    """Simple early stopping utility."""

    def __init__(self, patience: int = 8) -> None:
        self.best = float("inf")
        self.wait = 0
        self.patience = patience

    def __call__(self, current: float) -> bool:
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience


def save_checkpoint(model: torch.nn.Module, path: str | torch.PathLike) -> None:
    """Persist model weights."""
    torch.save(model.state_dict(), path)


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str) -> float:
    """Return average validation loss."""
    model.eval()
    total = 0.0
    crit = torch.nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            out = model(src, tgt[:, :-1])
            loss = crit(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
            total += loss.item()
    return total / max(1, len(loader))


def train_model(
    model: torch.nn.Module,
    dataset: Iterable[Any],
    cfg: Any,
    val_loader: DataLoader | None = None,
    ckpt_path: str | None = None,
) -> None:
    """Train ``model`` using ``dataset`` and configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if isinstance(dataset, IterableDataset):
        loader = DataLoader(
            dataset,
            batch_size=getattr(cfg, "batch_size", 1),
            shuffle=False,
            num_workers=min(2, getattr(cfg, "num_workers", 0)),
            pin_memory=False,
        )
        logging.getLogger(__name__).info(
            "prefetch ready in %.2fs", time.time() - t0
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=getattr(cfg, "batch_size", 1),
            shuffle=True,
            num_workers=getattr(cfg, "num_workers", 0),
            pin_memory=getattr(cfg, "pin_memory", False),
        )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=getattr(cfg, "learning_rate", 1e-3),
        weight_decay=getattr(cfg, "weight_decay", 0.0),
    )
    amp = bool(getattr(cfg, "use_mixed_precision", False)) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    early_stop = EarlyStopper(getattr(cfg, "early_stopping_patience", 8))
    crit = torch.nn.CrossEntropyLoss(ignore_index=0)
    model.to(device)
    for epoch in range(getattr(cfg, "num_epochs", getattr(cfg, "epochs", 1))):
        for step, batch in enumerate(loader, start=1):
            src, tgt = batch
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp):
                out = model(src, tgt[:, :-1])
                loss = crit(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            if getattr(cfg, "gradient_clipping", 0.0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(cfg, "gradient_clipping", 0.0))
            scaler.step(optimizer)
            scaler.update()
            if ckpt_path and getattr(cfg, "save_every", 0) and step % getattr(cfg, "save_every", 0) == 0:
                save_checkpoint(model, ckpt_path)
        if val_loader is not None:
            v_loss = evaluate(model, val_loader, device)
            if getattr(cfg, "early_stopping", False) and early_stop(v_loss):
                break
    if ckpt_path:
        save_checkpoint(model, ckpt_path)
