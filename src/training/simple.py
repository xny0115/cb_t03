from __future__ import annotations

from typing import List, Any, Dict, Tuple, Optional

import logging
import time
from pathlib import Path
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from ..data.loader import InstructionSample
from ..model.transformer import Seq2SeqTransformer
from ..utils.tokenizer import SentencePieceTokenizer
from .helpers import PairDataset, timed_collate, log_dataset_stats

logger = logging.getLogger(__name__)

def _prepare_dataset(
    samples: List[InstructionSample],
    tokenizer: SentencePieceTokenizer,
    is_pretrain: bool,
) -> Tuple[PairDataset, int]:
    """Encodes samples using a pre-trained tokenizer."""
    if is_pretrain:
        texts = [s.output for s in samples]
        log_dataset_stats(texts)

    pairs: List[Tuple[List[int], List[int]]] = []
    for s in samples:
        if is_pretrain:
            encoded = tokenizer.encode(s.output, add_special_tokens=True)
            src, tgt = encoded, encoded
        else:
            src_text = f"{s.instruction} {s.input}".strip()
            src = tokenizer.encode(src_text, add_special_tokens=True)
            tgt = tokenizer.encode(s.output, add_special_tokens=True)
        pairs.append((src, tgt))

    return PairDataset(pairs), len(samples)


def _create_loader(
    dataset: PairDataset, cfg: Dict[str, Any], *, drop_last: bool = True
) -> DataLoader:
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 0)) # Default to 0 for Windows compatibility
    pin_memory = bool(cfg.get("pin_memory", True))

    persistent_workers = num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=timed_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )
    return loader


def _init_model(
    tokenizer: SentencePieceTokenizer, cfg: Dict[str, Any]
) -> Seq2SeqTransformer:
    model_dim = int(cfg.get("model_dim", 256))
    num_heads = int(cfg.get("num_heads", 8))
    enc_layers = int(cfg.get("num_encoder_layers", 6))
    dec_layers = int(cfg.get("num_decoder_layers", 6))
    ff_dim = int(cfg.get("ff_dim", 1024))
    dropout = float(cfg.get("dropout_ratio", 0.1))

    model = Seq2SeqTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=model_dim,
        num_heads=num_heads,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dim_ff=ff_dim,
        dropout=dropout,
    )
    return model


def _train_epoch(
    loader: DataLoader,
    model: Seq2SeqTransformer,
    crit: nn.Module,
    opt: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: optim.lr_scheduler._LRScheduler,
    tokenizer: SentencePieceTokenizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    step_count = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        if tgt.size(1) < 2:
            continue

        opt.zero_grad(set_to_none=True)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(src, tgt_in, pad_id=tokenizer.pad_id)
            loss = crit(logits.view(-1, tokenizer.vocab_size), tgt_out.reshape(-1))

        if not torch.isfinite(loss):
            logger.warning(f"Warning: non-finite loss at step, skipping batch. Loss: {loss.item()}")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        scheduler.step()

        total_loss += loss.item()
        step_count += 1

    return total_loss / max(step_count, 1)


def _eval_epoch(
    loader: DataLoader,
    model: Seq2SeqTransformer,
    crit: nn.Module,
    tokenizer: SentencePieceTokenizer,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    step_count = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            with torch.cuda.amp.autocast(enabled=True):
                logits = model(src, tgt_in, pad_id=tokenizer.pad_id)
                loss = crit(logits.view(-1, tokenizer.vocab_size), tgt_out.reshape(-1))

            if torch.isfinite(loss):
                total_loss += loss.item()
                step_count += 1

    return total_loss / max(step_count, 1)


def train(
    samples: List[InstructionSample],
    cfg: dict[str, Any],
    *,
    is_pretrain: bool = False,
    model: Optional[Seq2SeqTransformer] = None,
) -> Seq2SeqTransformer:
    """Train a Seq2SeqTransformer on given samples. Can resume from an existing model."""

    tokenizer_path = cfg.get("tokenizer_path", "models/spm_bpe_8k.model")
    tokenizer_model_path = Path(tokenizer_path)
    if not tokenizer_model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found at {tokenizer_model_path}. Please run `scripts/prepare_data.py` first.")
    tokenizer = SentencePieceTokenizer(tokenizer_model_path)

    epochs = int(cfg.get("num_epochs", 5))

    dataset, line_count = _prepare_dataset(samples, tokenizer, is_pretrain)

    train_size = int(0.95 * len(dataset))
    if train_size < 1:
        raise ValueError("Dataset is too small to create a training set.")
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    loader = _create_loader(train_set, cfg, drop_last=True)
    val_loader = _create_loader(val_set, cfg, drop_last=False)

    if model is None:
        logger.info("No existing model provided, initializing a new one.")
        model = _init_model(tokenizer, cfg)
    else:
        logger.info("Resuming training with the existing model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("CUDA unavailable. A GPU environment is required for training.")

    logger.info(f"Initializing model on device: {device}")
    model.to(device)

    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    opt = optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("use_mixed_precision", True)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(loader) * epochs)

    logger.info(f"Training start: epochs={epochs}, samples={line_count}, vocab_size={tokenizer.vocab_size}")

    train_start = time.perf_counter()
    best_val_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = int(cfg.get("early_stopping_patience", 3))
    checkpoint_path = Path("models/best_model_checkpoint.pth")

    # If resuming, check current validation loss
    if Path.exists(checkpoint_path):
        try:
            # Make sure to load to the correct device
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            best_val_loss = _eval_epoch(val_loader, model, crit, tokenizer, device)
            logger.info(f"Resuming from checkpoint. Initial validation loss: {best_val_loss:.4f}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint, starting fresh. Error: {e}")


    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        train_loss = _train_epoch(loader, model, crit, opt, scaler, scheduler, tokenizer, device)
        val_loss = _eval_epoch(val_loader, model, crit, tokenizer, device)

        epoch_duration = time.perf_counter() - epoch_start

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_duration:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"New best model saved to {checkpoint_path} with validation loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    total_duration = time.perf_counter() - train_start
    logger.info(f"Training complete in {total_duration:.2f}s")

    if checkpoint_path.exists():
        logger.info(f"Loading best model from checkpoint.")
        model.load_state_dict(torch.load(checkpoint_path))

    return model

def pretrain(
    texts: List[str],
    cfg: dict[str, Any] | None = None,
    model: Optional[Seq2SeqTransformer] = None
) -> Seq2SeqTransformer:
    """Pre-training wrapper."""
    cfg = cfg or {}
    samples = [InstructionSample(instruction="", input="", output=t) for t in texts]
    return train(samples, cfg, is_pretrain=True, model=model)
