"""CPU fallback training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any, Iterable
import json
from datetime import datetime
import time
import logging

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .data.loader import QADataset
from .utils.vocab import build_vocab, encode_tokens
from .model.transformer import Seq2SeqTransformer
from .trainer import EarlyStopper, evaluate


class TorchQADataset(Dataset):
    """PyTorch dataset wrapper."""

    def __init__(self, data: QADataset, vocab: dict[str, int]) -> None:
        if not vocab:
            raise ValueError("empty vocab")
        self.data = data
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        pair = self.data.pairs[idx]
        if not pair.tokens_q or not pair.tokens_a:
            raise ValueError("invalid pair")
        q = encode_tokens(pair.tokens_q, pair.concepts, pair.domain, self.vocab)
        a = encode_tokens(pair.tokens_a, pair.concepts, pair.domain, self.vocab)
        return q, a


def collate_fn(batch: Iterable[tuple[Any, Any]]):
    qs, as_ = zip(*batch)
    qs = nn.utils.rnn.pad_sequence(qs, batch_first=True, padding_value=0)
    as_ = nn.utils.rnn.pad_sequence(as_, batch_first=True, padding_value=0)
    return qs, as_


def _train_epoch_cpu(
    loader: DataLoader,
    model: nn.Module,
    crit: nn.Module,
    opt: optim.Optimizer,
    vocab_size: int,
    grad_clip: float,
    interval: int = 0,
    save_path: Path | None = None,
) -> float:
    model.train()
    total = 0.0
    for step, (src, tgt) in enumerate(loader, start=1):
        opt.zero_grad()
        out = model(src, tgt[:, :-1])
        loss = crit(out.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
        loss.backward()
        if grad_clip > 0:
            if step == 1:
                logger.info("Grad clip %.1f", grad_clip)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if interval and save_path and step % interval == 0:
            torch.save(model.state_dict(), save_path)
        total += loss.item()
    return total / len(loader)
logger = logging.getLogger(__name__)
def train_cpu(
    dataset: QADataset,
    cfg: Config,
    save_path: Path,
    start_epoch: int,
    meta_path: Path | None,
    progress_cb: Callable | None,
) -> Path:
    """Run simplified CPU training loop."""

    vocab = build_vocab(dataset)
    full_ds = TorchQADataset(dataset, vocab)
    val_count = max(1, int(len(full_ds) * 0.1))
    train_count = len(full_ds) - val_count
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_count, val_count])
    loader = DataLoader(
        train_ds,
        batch_size=max(cfg.batch_size, 1),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(cfg.batch_size, 1),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    model = Seq2SeqTransformer(vocab_size=len(vocab))
    if start_epoch and save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location="cpu"))

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    stopper = EarlyStopper(cfg.early_stopping_patience)
    for e in range(start_epoch, cfg.num_epochs):
        st = time.perf_counter()
        loss = _train_epoch_cpu(
            loader,
            model,
            criterion,
            optimizer,
            len(vocab),
            cfg.gradient_clipping,
            cfg.save_every,
            save_path if cfg.save_every else None,
        )
        val_loss = evaluate(model, val_loader, "cpu")
        if progress_cb:
            progress_cb(e + 1, cfg.num_epochs, val_loss)
        if meta_path:
            json.dump(
                {"epochs_done": e + 1, "update_time": datetime.utcnow().isoformat()},
                open(meta_path, "w"),
            )
        if not cfg.save_every:
            torch.save(model.state_dict(), save_path)
        elapsed = (time.perf_counter() - st) * 1000.0
        logger.info(
            "epoch %d/%d | loss=%.4f | val=%.4f | %.0fms",
            e + 1,
            cfg.num_epochs,
            loss,
            val_loss,
            elapsed,
        )
        if cfg.early_stopping and stopper(val_loss):
            logger.info("early stopping triggered")
            break

    vocab_path = save_path.with_suffix(".vocab.json")
    json.dump(vocab, open(vocab_path, "w"), ensure_ascii=False)
    return save_path

