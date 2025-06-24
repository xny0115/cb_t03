from __future__ import annotations

from typing import List, Tuple, Any

import logging
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from ..data.loader import InstructionSample
from ..model.transformer import Seq2SeqTransformer
from ..utils.tokenizer import CharTokenizer


class _PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[List[int], List[int]]]):
        self.data = pairs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def _collate(batch):
    srcs, tgts = zip(*batch)
    srcs = [torch.tensor(s, dtype=torch.long) for s in srcs]
    tgts = [torch.tensor(t, dtype=torch.long) for t in tgts]
    src_pad = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    return src_pad, tgt_pad


def train(samples: List[InstructionSample], cfg: dict[str, Any] | None = None):
    """Train a Seq2SeqTransformer on given samples."""

    logger = logging.getLogger(__name__)
    cfg = cfg or {}
    epochs = int(cfg.get("num_epochs", 5))
    lr = float(cfg.get("learning_rate", 1e-3))
    batch_size = int(cfg.get("batch_size", 8))
    model_dim = int(cfg.get("model_dim", 128))
    num_heads = int(cfg.get("num_heads", 4))
    enc_layers = int(cfg.get("num_encoder_layers", 2))
    dec_layers = int(cfg.get("num_decoder_layers", 2))
    ff_dim = int(cfg.get("ff_dim", 512))
    dropout = float(cfg.get("dropout_ratio", 0.1))
    texts = [f"{s.instruction} {s.input} {s.output}" for s in samples]
    tokenizer = CharTokenizer(texts)
    pairs = []
    for s in samples:
        src = tokenizer.encode(f"{s.instruction} {s.input}".strip(), True)
        tgt = tokenizer.encode(s.output, True)
        pairs.append((src, tgt))

    dataset = _PairDataset(pairs)
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False)) and torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = Seq2SeqTransformer(
        tokenizer.vocab_size,
        embed_dim=model_dim,
        num_heads=num_heads,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dim_ff=ff_dim,
        dropout=dropout,
    )
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
        logger.warning(
            "CUDA not available, falling back to CPU. Training will be slow"
        )
    logger.info("Training device: %s", device)
    model.to(device)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.Adam(model.parameters(), lr=lr)
    use_amp = bool(cfg.get("use_mixed_precision", False)) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger.info("Training start: epochs=%d, samples=%d", epochs, len(samples))
    train_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start = time.perf_counter()
        for src, tgt in loader:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(src, tgt[:, :-1])
                loss = crit(out.reshape(-1, tokenizer.vocab_size), tgt[:, 1:].reshape(-1))
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            total_loss += loss.item()
        ms = time.perf_counter() - start
        avg_loss = total_loss / len(loader)
        progress = ((epoch + 1) / epochs) * 100
        logger.info(
            "Epoch %d/%d (%.0f%%) | Loss: %.3f | Time: %.2fs",
            epoch + 1,
            epochs,
            progress,
            avg_loss,
            ms,
        )

    logger.info("Training complete in %.2fs", time.perf_counter() - train_start)

    return model, tokenizer


def pretrain(texts: List[str], cfg: dict[str, Any] | None = None):
    """사전학습용 간단한 오토인코더 방식."""
    samples = [InstructionSample("", "", t) for t in texts]
    return train(samples, cfg)
