from __future__ import annotations

from typing import List, Any, Dict, Tuple

import logging
import time
import warnings
import platform
import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from ..data.loader import InstructionSample
from ..model.transformer import Seq2SeqTransformer, save_transformer
from ..utils.tokenizer import CharTokenizer
from .helpers import PairDataset, timed_collate, log_dataset_stats

logger = logging.getLogger(__name__)

def _prepare_dataset(
    samples: List[InstructionSample], is_pretrain: bool
) -> Tuple[PairDataset, CharTokenizer, int]:
    if is_pretrain:
        texts = [s.output for s in samples]
    else:
        texts = [f"{s.instruction} {s.input} {s.output}" for s in samples]
    log_dataset_stats(texts)
    start = time.perf_counter()
    tokenizer = CharTokenizer(texts)
    logger.debug("tokenizer build time: %.2fs", time.perf_counter() - start)

    pairs: List[Tuple[List[int], List[int]]] = []
    encode_times: List[float] = []
    for idx, s in enumerate(samples):
        if is_pretrain:
            src = tokenizer.encode(s.output, True)
            tgt = src
        else:
            src = tokenizer.encode(f"{s.instruction} {s.input}".strip(), True)
            tgt = tokenizer.encode(s.output, True)
        if idx < 10:
            encode_times.append((time.perf_counter() - start) * 1000)
            logger.debug("encode sample %d: %s", idx, s.output[:50])
        pairs.append((src, tgt))
    if encode_times:
        logger.debug("avg encode time: %.2fms", sum(encode_times) / len(encode_times))

    return PairDataset(pairs), tokenizer, len(samples)


def _create_loader(dataset: PairDataset, cfg: Dict[str, Any]) -> DataLoader:
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", min(max(os.cpu_count() // 2, 2), 8)))
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
    logging.getLogger(__name__).debug(
        "dataloader build time: %.2fs", time.perf_counter() - start
    )
    return loader


def _init_model(
    tokenizer: CharTokenizer, cfg: Dict[str, Any]
) -> Tuple[
    Seq2SeqTransformer, nn.Module, optim.Optimizer, torch.cuda.amp.GradScaler, str, bool
]:
    model_dim = int(cfg.get("model_dim", 128))
    num_heads = int(cfg.get("num_heads", 4))
    enc_layers = int(cfg.get("num_encoder_layers", 2))
    dec_layers = int(cfg.get("num_decoder_layers", 2))
    ff_dim = int(cfg.get("ff_dim", 512))
    dropout = float(cfg.get("dropout_ratio", 0.1))
    model = Seq2SeqTransformer(
        tokenizer.vocab_size,
        embed_dim=model_dim,
        num_heads=num_heads,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dim_ff=ff_dim,
        dropout=dropout,
    )
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. GPU 환경이 필요합니다.")
    torch.backends.cudnn.benchmark = True
    model.to(device)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    use_amp = bool(cfg.get("use_mixed_precision", False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    return model, crit, opt, scaler, device, use_amp


def _train_epoch(
    loader: DataLoader,
    model: Seq2SeqTransformer,
    crit: nn.Module,
    opt: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: optim.lr_scheduler._LRScheduler,
    tokenizer: CharTokenizer,
    device: str,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    skipped_batch = 0
    skipped_pad = 0
    step_count = 0
    start = time.perf_counter()
    for i, (src, tgt) in enumerate(loader):
        src, tgt = src.to(device), tgt.to(device)
        if tgt.size(1) < 2:
            skipped_batch += 1
            continue
        opt.zero_grad()
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        with torch.cuda.amp.autocast(enabled=use_amp):
            # --- NaN 예방: 유효 토큰이 하나도 없으면 배치 건너뜀 ---
            if (tgt_out != tokenizer.pad_id).sum() == 0:
                skipped_pad += 1
                continue
            logits = model(src, tgt_in, tokenizer.pad_id)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                tgt_out.flatten(0, 1),
                ignore_index=tokenizer.pad_id,
            )
        if not torch.isfinite(loss):
            logger.error(
                "Loss became non-finite. Disabling AMP and restarting in float32."
            )
            use_amp = False
            scaler = torch.cuda.amp.GradScaler(enabled=False)
            continue
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()
        total_loss += loss.item()
        step_count += 1
    duration = time.perf_counter() - start
    # average loss per batch
    avg_loss = total_loss / max(step_count, 1)
    logger.info("batches skipped (too short): %d", skipped_batch)
    logger.info("batches skipped (pad-only): %d", skipped_pad)
    return avg_loss, duration


def _eval_epoch(
    loader: DataLoader,
    model: Seq2SeqTransformer,
    crit: nn.Module,
    tokenizer: CharTokenizer,
    device: str,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (src, tgt) in enumerate(loader):
            src, tgt = src.to(device), tgt.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(src, tgt[:, :-1], tokenizer.pad_id)
                loss = F.cross_entropy(
                    out.reshape(-1, tokenizer.vocab_size),
                    tgt[:, 1:].reshape(-1),
                    ignore_index=tokenizer.pad_id,
                )
            total_loss += loss.item()
    return total_loss / (i + 1)


def train(
    samples: List[InstructionSample],
    cfg: dict[str, Any] | None = None,
    *,
    is_pretrain: bool = False,
    save_dir: str | None = None,
) -> Tuple[Seq2SeqTransformer, CharTokenizer]:
    """Train a Seq2SeqTransformer on given samples."""
    torch.autograd.set_detect_anomaly(True)
    if platform.system() == "Windows":
        warnings.filterwarnings(
            "ignore",
            message="Torch was not compiled with flash attention",
            category=UserWarning,
        )
    cfg = cfg or {}
    epochs = int(cfg.get("num_epochs", 5))

    dataset, tokenizer, line_count = _prepare_dataset(samples, is_pretrain)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    loader = _create_loader(train_set, cfg)
    val_loader = _create_loader(val_set, cfg)
    model, crit, opt, scaler, device, use_amp = _init_model(tokenizer, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    logger.info("Training start: epochs=%d, samples=%d", epochs, line_count)
    param_rates: List[float] = []
    snapshot = lambda m: {
        k: v.detach().clone().cpu() for k, v in m.state_dict().items()
    }
    prev_state = snapshot(model)
    prev_ratio = None
    train_start = time.perf_counter()
    best = float("inf")
    patience, counter = 3, 0
    best_state = snapshot(model)
    for epoch in range(epochs):
        loss, duration = _train_epoch(
            loader, model, crit, opt, scaler, scheduler, tokenizer, device, use_amp
        )
        val_loss = _eval_epoch(val_loader, model, crit, tokenizer, device, use_amp)
        ratio = duration / max(line_count, 1)
        if prev_ratio is not None:
            ratio_drop = prev_ratio - ratio
            if abs(ratio_drop) < 0.05:
                logger.info(
                    "epoch %d time/line ratio changed: %.4f -> %.4f",
                    epoch + 1,
                    prev_ratio,
                    ratio,
                )
            else:
                logger.warning(
                    "epoch %d time/line ratio dropped: %.4f -> %.4f",
                    epoch + 1,
                    prev_ratio,
                    ratio,
                )
        prev_ratio = ratio
        curr = snapshot(model)
        changed = sum(not torch.equal(curr[k], prev_state[k]) for k in curr)
        rate = changed / len(curr) * 100
        if len(curr) - changed >= len(curr) * 0.5:
            logger.warning(
                "epoch %d: %d/%d params unchanged",
                epoch + 1,
                len(curr) - changed,
                len(curr),
            )
        param_rates.append(rate)
        prev_state = snapshot(model)
        logger.info(
            "Epoch %d/%d | Loss: %.3f | Val: %.3f | Time: %.2fs",
            epoch + 1,
            epochs,
            loss,
            val_loss,
            duration,
        )
        if val_loss < best * 0.995:
            best, counter = val_loss, 0
            best_state = snapshot(model)
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Training complete in %.2fs", time.perf_counter() - train_start)
    for idx, r in enumerate(param_rates, 1):
        logger.debug("Epoch %d → %.1f%% 변화", idx, r)
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model_path = save_path / "model.pth"
        if counter >= patience:
            model.load_state_dict(best_state)
        save_transformer(model, tokenizer.stoi, model_path)
        tokenizer.save(save_path / "tokenizer.json")
        logger.info("Model & tokenizer saved to %s", save_dir)
    return model, tokenizer


def pretrain(texts: List[str], cfg: dict[str, Any] | None = None):
    """사전학습용 간단한 오토인코더 방식."""
    samples = [InstructionSample("", "", t) for t in texts]
    return train(samples, cfg, is_pretrain=True)
