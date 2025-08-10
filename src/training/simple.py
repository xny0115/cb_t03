from __future__ import annotations

from typing import List, Any, Dict, Tuple

import logging
import time
import warnings
import platform
from pathlib import Path
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from ..data.loader import InstructionSample
from ..model.transformer import Seq2SeqTransformer, save_transformer
from ..utils.tokenizer import SentencePieceTokenizer
from .helpers import PairDataset, timed_collate, log_dataset_stats

logger = logging.getLogger(__name__)

def _prepare_dataset(
    samples: List[InstructionSample], tokenizer: SentencePieceTokenizer, is_pretrain: bool
) -> Tuple[PairDataset, int]:
    if is_pretrain:
        texts = [s.output for s in samples]
    else:
        texts = [f"{s.instruction} {s.input} {s.output}" for s in samples]
    log_dataset_stats(texts)

    pairs: List[Tuple[List[int], List[int]]] = []
    for idx, s in enumerate(samples):
        if is_pretrain:
            src = tokenizer.encode(s.output, True)
            tgt = src
        else:
            src = tokenizer.encode(f"{s.instruction} {s.input}".strip(), True)
            tgt = tokenizer.encode(s.output, True)
        pairs.append((src, tgt))

    return PairDataset(pairs), len(samples)


def _create_loader(
    dataset: PairDataset, cfg: Dict[str, Any], *, drop_last: bool = True
) -> DataLoader:
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 4))
    pin_memory = bool(cfg.get("pin_memory", True))
    start = time.perf_counter()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=timed_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=True,
    )
    logging.getLogger(__name__).debug(
        "dataloader build time: %.2fs", time.perf_counter() - start
    )
    return loader


def _init_model(
    tokenizer: SentencePieceTokenizer, cfg: Dict[str, Any]
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("CUDA unavailable. GPU 환경이 필요합니다.")
    model.to(device)

    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"✅✅✅ MODEL LOADED ON GPU: {gpu_name} ✅✅✅")

    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    opt = optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    amp_enabled = bool(cfg.get("use_mixed_precision", False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    return model, crit, opt, scaler, device, amp_enabled


def _train_epoch(
    loader: DataLoader,
    model: Seq2SeqTransformer,
    crit: nn.Module,
    opt: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: optim.lr_scheduler._LRScheduler,
    tokenizer: SentencePieceTokenizer,
    device: str,
    amp_enabled: bool,
) -> Tuple[float, float, torch.cuda.amp.GradScaler, bool]:
    model.train()
    total_loss = 0.0
    step_count = 0
    start = time.perf_counter()
    for i, (src, tgt) in enumerate(loader):
        src, tgt = src.to(device), tgt.to(device)
        if tgt.size(1) < 2:
            continue
        opt.zero_grad()
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            if (tgt_out != tokenizer.pad_id).sum() == 0:
                continue
            logits = model(src, tgt_in, tokenizer.pad_id)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                tgt_out.flatten(0, 1),
                ignore_index=tokenizer.pad_id,
            )
        if not torch.isfinite(loss):
            logger.error("non-finite loss at step %d, AMP=%s", i, amp_enabled)
            if amp_enabled:
                amp_enabled = False
                scaler = torch.cuda.amp.GradScaler(enabled=False)
            else:
                raise RuntimeError("FP32 NaN detected. Training stopped.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        step_count += 1
    duration = time.perf_counter() - start
    avg_loss = total_loss / max(step_count, 1)
    return avg_loss, duration, scaler, amp_enabled


def _eval_epoch(
    loader: DataLoader,
    model: Seq2SeqTransformer,
    crit: nn.Module,
    tokenizer: SentencePieceTokenizer,
    device: str,
    amp_enabled: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    step_count = 0
    with torch.no_grad():
        for i, (src, tgt) in enumerate(loader):
            src, tgt = src.to(device), tgt.to(device)
            if tgt.size(1) < 2:
                continue
            step_count +=1
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(src, tgt[:, :-1], tokenizer.pad_id)
                loss = F.cross_entropy(
                    out.reshape(-1, tokenizer.vocab_size),
                    tgt[:, 1:].reshape(-1),
                    ignore_index=tokenizer.pad_id,
                )
            total_loss += loss.item()
    return total_loss / max(step_count, 1)


def train(
    samples: List[InstructionSample],
    cfg: dict[str, Any] | None = None,
    *,
    is_pretrain: bool = False,
    save_dir: str | None = None,
) -> Tuple[Seq2SeqTransformer, SentencePieceTokenizer]:
    """Train a Seq2SeqTransformer on given samples."""
    torch.autograd.set_detect_anomaly(True)
    if platform.system() == "Windows":
        warnings.filterwarnings(
            "ignore",
            message="Torch was not compiled with flash attention",
            category=UserWarning,
        )
    cfg = cfg or {}

    spm_model_path = str(cfg.get("spm_model_path", "tokenizer/spm.model"))
    if not Path(spm_model_path).exists():
        raise FileNotFoundError(f"SentencePiece model not found: {spm_model_path}")
    tokenizer = SentencePieceTokenizer(spm_model_path)

    model, crit, opt, scaler, device, amp_enabled = _init_model(tokenizer, cfg)
    epochs = int(cfg.get("num_epochs", 5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch = 0

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    resume = bool(cfg.get("resume", False))
    if resume and save_dir:
        checkpoint_path = Path(save_dir) / "checkpoint.pth"
        if checkpoint_path.exists():
            logger.info("Loading checkpoint from %s for resuming training", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            logger.info("Resuming from epoch %d", start_epoch)
        else:
            logger.warning("Resume mode is on, but checkpoint not found. Starting new training: %s", checkpoint_path)

    dataset, line_count = _prepare_dataset(samples, tokenizer, is_pretrain)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    loader = _create_loader(train_set, cfg, drop_last=True)
    val_loader = _create_loader(val_set, cfg, drop_last=False)

    logger.info("Training start: epochs=%d, samples=%d, start_epoch=%d", epochs, line_count, start_epoch)

    train_start = time.perf_counter()
    best_val_loss = float("inf")
    patience, counter = int(cfg.get("early_stopping_patience", 3)), 0

    final_epoch = 0

    for epoch in range(start_epoch, epochs):
        final_epoch = epoch
        loss, duration, scaler, amp_enabled = _train_epoch(
            loader, model, crit, opt, scaler, scheduler, tokenizer, device, amp_enabled
        )
        val_loss = _eval_epoch(val_loader, model, crit, tokenizer, device, amp_enabled)

        logger.info(
            "Epoch %d/%d | Loss: %.3f | Val: %.3f | Time: %.2fs",
            epoch + 1,
            epochs,
            loss,
            val_loss,
            duration,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # No model cloning or saving inside the loop for performance.
            # The best model will be loaded from the final checkpoint.
            logger.info("New best validation loss: %.3f. Checkpoint will be saved at the end.", best_val_loss)
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Training complete in %.2fs", time.perf_counter() - train_start)

    if save_dir:
        save_path = Path(save_dir)

        # Save final model for inference (using the state from the last epoch)
        model_path = save_path / "model.pth"
        save_transformer(model, {}, model_path)

        # Save final checkpoint
        checkpoint_path = save_path / "checkpoint.pth"
        torch.save({
            'epoch': final_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': best_val_loss,
        }, checkpoint_path)
        logger.info("Final checkpoint for model from epoch %d saved to %s", final_epoch + 1, checkpoint_path)

    return model, tokenizer


def pretrain(texts: List[str], cfg: dict[str, Any] | None = None, save_dir: str | None = None):
    """사전학습용 간단한 오토인코더 방식."""
    samples = [InstructionSample("", "", t) for t in texts]
    return train(samples, cfg, is_pretrain=True, save_dir=save_dir)
