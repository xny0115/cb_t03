from __future__ import annotations

from typing import List, Any

import logging
import time
import warnings
import platform

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ..data.loader import InstructionSample
from ..model.transformer import Seq2SeqTransformer
from ..utils.tokenizer import CharTokenizer
from .helpers import PairDataset, timed_collate, log_dataset_stats


def train(
    samples: List[InstructionSample],
    cfg: dict[str, Any] | None = None,
    *,
    is_pretrain: bool = False,
):
    """Train a Seq2SeqTransformer on given samples."""

    logger = logging.getLogger(__name__)
    if platform.system() == "Windows":
        warnings.filterwarnings(
            "ignore",
            message="Torch was not compiled with flash attention",
            category=UserWarning,
        )
    cfg = cfg or {}
    epochs = int(cfg.get("num_epochs", 5))
    lr = float(cfg.get("learning_rate", 1e-3))
    model_dim = int(cfg.get("model_dim", 128))
    num_heads = int(cfg.get("num_heads", 4))
    enc_layers = int(cfg.get("num_encoder_layers", 2))
    dec_layers = int(cfg.get("num_decoder_layers", 2))
    ff_dim = int(cfg.get("ff_dim", 512))
    dropout = float(cfg.get("dropout_ratio", 0.1))
    if is_pretrain:
        texts = [s.output for s in samples]
    else:
        texts = [f"{s.instruction} {s.input} {s.output}" for s in samples]
    log_dataset_stats(texts)
    start = time.perf_counter()
    tokenizer = CharTokenizer(texts)
    logger.debug("tokenizer build time: %.2fs", time.perf_counter() - start)
    pairs = []
    src_len_sum = 0
    tgt_len_sum = 0
    encode_times: List[float] = []
    example: str | None = None
    for idx, s in enumerate(samples):
        if is_pretrain:
            input_text = s.output
            t0 = time.perf_counter()
            src = tokenizer.encode(input_text, True)
            tgt = src
        else:
            input_text = f"{s.instruction}{s.input}{s.output}"
            t0 = time.perf_counter()
            src = tokenizer.encode(f"{s.instruction} {s.input}".strip(), True)
            tgt = tokenizer.encode(s.output, True)
        if len(encode_times) < 10:
            encode_times.append((time.perf_counter() - t0) * 1000)
            if example is None:
                example = input_text
        if idx < 2:
            logger.debug("encode sample %d: %s", idx, input_text[:50])
        src_len_sum += len(src)
        tgt_len_sum += len(tgt)
        pairs.append((src, tgt))
    if encode_times:
        logger.debug(
            "avg encode time: %.2fms", sum(encode_times) / len(encode_times)
        )
        logger.debug("encode example: %s", example[:50] if example else "")

    dataset = PairDataset(pairs)
    if samples:
        avg_instr = sum(len(s.instruction) for s in samples) / len(samples)
        avg_input = sum(len(s.input) for s in samples) / len(samples)
        avg_output = sum(len(s.output) for s in samples) / len(samples)
        avg_src_tok = src_len_sum / len(samples)
        avg_tgt_tok = tgt_len_sum / len(samples)
        logger.debug("avg instruction length: %.2f", avg_instr)
        logger.debug("avg input length: %.2f", avg_input)
        logger.debug("avg output length: %.2f", avg_output)
        logger.debug("avg src tokens: %.2f", avg_src_tok)
        logger.debug("avg tgt tokens: %.2f", avg_tgt_tok)
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(
        cfg.get("num_workers", min(max(os.cpu_count() // 2, 2), 8))
    )
    pin_memory = bool(cfg.get("pin_memory", True))
    start = time.perf_counter()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=timed_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    logger.debug("dataloader build time: %.2fs", time.perf_counter() - start)
    iter_start = time.perf_counter()
    _ = iter(loader)
    logger.debug(
        "dataloader iter build time: %.2fms", (time.perf_counter() - iter_start) * 1000
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
    if tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 1):
        pass  # compile removed for Windows compatibility
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
    logger.debug("model device: %s", next(model.parameters()).device)
    if device == "cuda":
        logger.debug(
            "CUDA max memory allocated: %.2fMB",
            torch.cuda.max_memory_allocated() / 1_048_576,
        )
        logger.debug(
            "allocated memory: %.2f MB",
            torch.cuda.memory_allocated() / 1024**2,
        )
        logger.debug(
            "reserved memory: %.2f MB",
            torch.cuda.memory_reserved() / 1024**2,
        )
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.Adam(model.parameters(), lr=lr)
    use_amp = bool(cfg.get("use_mixed_precision", False)) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger.info("Training start: epochs=%d, samples=%d", epochs, len(samples))
    logger.info(
        "Config: batch_size=%d, model_dim=%d, ff_dim=%d, enc_layers=%d, dec_layers=%d, lr=%.4f, dropout=%.2f, mixed_precision=%s",
        batch_size,
        model_dim,
        ff_dim,
        enc_layers,
        dec_layers,
        lr,
        dropout,
        use_amp,
    )
    train_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        epoch_start = time.perf_counter()
        for i, (src, tgt) in enumerate(loader):
            move_start = time.perf_counter()
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            move_ms = (time.perf_counter() - move_start) * 1000
            if i < 2:
                logger.debug(
                    "batch %d src%s tgt%s dtype=%s device=%s move=%.2fms",
                    i,
                    tuple(src.shape),
                    tuple(tgt.shape),
                    src.dtype,
                    src.device,
                    move_ms,
                )
            opt.zero_grad()
            fwd_start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(src, tgt[:, :-1])
                loss = crit(out.reshape(-1, tokenizer.vocab_size), tgt[:, 1:].reshape(-1))
            fwd_ms = (time.perf_counter() - fwd_start) * 1000
            bwd_start = time.perf_counter()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            bwd_ms = (time.perf_counter() - bwd_start) * 1000
            total_loss += loss.item()
            if i < 2:
                logger.debug(
                    "batch %d forward=%.2fms backward=%.2fms", i, fwd_ms, bwd_ms
                )
            logger.debug(
                "batch %d total: %.2fms", i, move_ms + fwd_ms + bwd_ms
            )
            if i >= 2:
                break
        ms = time.perf_counter() - epoch_start
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
        logger.debug("epoch %d time: %.2fs", epoch + 1, ms)

    logger.info("Training complete in %.2fs", time.perf_counter() - train_start)
    logger.debug("epoch execution time: %.2fs", time.perf_counter() - train_start)

    return model, tokenizer


def pretrain(texts: List[str], cfg: dict[str, Any] | None = None):
    """사전학습용 간단한 오토인코더 방식."""
    samples = [InstructionSample("", "", t) for t in texts]
    return train(samples, cfg, is_pretrain=True)
