from __future__ import annotations

from typing import List, Any, Dict, Tuple, Optional
import logging
import time
import warnings
import platform
from pathlib import Path
import os
import torch

# Performance settings from GPT instructions
from torch.backends.cuda import sdp_kernel
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass # Older torch versions may not have this
sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
torch.backends.cudnn.benchmark = True


from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from ..data.loader import InstructionSample
from ..model.transformer import Seq2SeqTransformer, save_transformer, load_transformer
from ..utils.tokenizer import SentencePieceTokenizer
from .helpers import PairDataset, collate, timed_collate, log_dataset_stats
from .checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

def _prepare_dataset(
    samples: List[InstructionSample], tokenizer: SentencePieceTokenizer, is_pretrain: bool
) -> Tuple[PairDataset, int]:
    if is_pretrain:
        texts = [s.output for s in samples]
    else:
        texts = [f"{s.instruction} {s.input} {s.output}" for s in samples]

    pairs: List[Tuple[List[int], List[int]]] = []
    for s in samples:
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
    num_workers_default = max(2, os.cpu_count() // 2) if platform.system() != "Windows" else 0
    num_workers = int(cfg.get("num_workers", num_workers_default))
    batch_size = int(cfg.get("batch_size", 128))
    pin_memory = bool(cfg.get("pin_memory", True))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader


def _init_model(
    tokenizer: SentencePieceTokenizer, cfg: Dict[str, Any]
) -> Tuple[
    Seq2SeqTransformer, nn.Module, optim.Optimizer, torch.cuda.amp.GradScaler, str, bool
]:
    model_dim = int(cfg.get("model_dim", 256))
    num_heads = int(cfg.get("num_heads", 8))
    enc_layers = int(cfg.get("num_encoder_layers", 6))
    dec_layers = int(cfg.get("num_decoder_layers", 6))
    ff_dim = int(cfg.get("ff_dim", 1024))
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
        if os.getenv("ALLOW_CPU_TRAINING") == "1":
            logger.warning("CUDA not available: training on CPU (slow).")
        else:
            raise RuntimeError("CUDA required")
    model.to(device)

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✅✅✅ MODEL LOADED ON GPU: {gpu_name} ✅✅✅")

    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    opt = optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-4)))
    amp_enabled = bool(cfg.get("use_mixed_precision", True)) and (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    return model, crit, opt, scaler, device, amp_enabled


def _train_epoch(
    loader: DataLoader,
    model: Seq2SeqTransformer,
    crit: nn.Module,
    opt: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    tokenizer: SentencePieceTokenizer,
    device: str,
    amp_enabled: bool,
) -> Tuple[float, float, float, float]:
    model.train()
    total_loss = 0.0
    step_count = 0
    start_time = time.perf_counter()
    for i, (src, tgt) in enumerate(loader):
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        if tgt.size(1) < 2:
            continue
        try:
            opt.zero_grad(set_to_none=True)
        except TypeError:
            opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(src, tgt[:, :-1], tokenizer.pad_id)
            loss = crit(logits.flatten(0, 1), tgt[:, 1:].flatten(0, 1))

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()
        step_count += 1

    duration = time.perf_counter() - start_time
    avg_loss = total_loss / max(step_count, 1)
    if scheduler:
        scheduler.step()
    return avg_loss, duration, 0.0, 0.0


def train(
    samples: List[InstructionSample],
    cfg: dict[str, Any] | None = None,
    *,
    is_pretrain: bool = False,
    save_dir: str | None = None,
) -> Tuple[Seq2SeqTransformer, SentencePieceTokenizer]:
    cfg = cfg or {}

    spm_model_path = str(cfg.get("spm_model_path", "tokenizer/spm.model"))
    if not Path(spm_model_path).exists():
        raise FileNotFoundError(
            f"SentencePiece model not found: {spm_model_path}. "
            f"Run: python train_spm.py --input \"datas/pretrain/**/*.txt\""
        )
    tokenizer = SentencePieceTokenizer(spm_model_path)

    model, crit, opt, scaler, device, amp_enabled = _init_model(tokenizer, cfg)
    epochs = int(cfg.get("num_epochs", 5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch = 0

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if cfg.get("resume", False):
            mode_name = 'pretrain' if is_pretrain else 'finetune'
            last_ckpt_path = save_dir / f"last_{mode_name}.ckpt"
            best_model_path = save_dir / f"{mode_name}.pth"

            if last_ckpt_path.exists():
                checkpoint = load_checkpoint(last_ckpt_path, model, opt, scheduler, scaler)
                start_epoch = checkpoint.get('epoch', -1) + 1
                logger.info("Resuming training from last checkpoint: %s (epoch %d)", last_ckpt_path, start_epoch)
            elif best_model_path.exists():
                logger.info("Last checkpoint not found. Resuming from best model weights: %s", best_model_path)
                loaded_model, _ = load_transformer(best_model_path)
                model.load_state_dict(loaded_model.state_dict())
                logger.info("Optimizer and scheduler are reset.")
            else:
                logger.warning("Resume mode is on, but no checkpoint or model found. Starting new training.")


    dataset, line_count = _prepare_dataset(samples, tokenizer, is_pretrain)
    loader = _create_loader(dataset, cfg, drop_last=True)

    logger.info("Training start: epochs=%d, samples=%d, start_epoch=%d", epochs, line_count, start_epoch)
    train_start = time.perf_counter()
    final_epoch = 0

    for epoch in range(start_epoch, epochs):
        final_epoch = epoch
        loss, duration, _, _ = _train_epoch(
            loader, model, crit, opt, scaler, scheduler, tokenizer, device, amp_enabled
        )

        logger.info(
            "Epoch %d/%d | Train Loss: %.3f | Time: %.2fs",
            epoch + 1, epochs, loss, duration,
        )

        if save_dir:
            last_ckpt_path = save_dir / f"last_{'pretrain' if is_pretrain else 'finetune'}.ckpt"
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config_snapshot': cfg,
                'tokenizer_info': {"type": "spm", "spm_model_path": spm_model_path}
            }
            save_checkpoint(state, last_ckpt_path)

    logger.info("Training complete in %.2fs", time.perf_counter() - train_start)

    if save_dir:
        model_path = save_dir / f"{'pretrain' if is_pretrain else 'finetune'}.pth"
        save_transformer(model, {}, model_path)

    return model, tokenizer


def pretrain(texts: List[str], cfg: dict[str, Any] | None = None, save_dir: str | None = None):
    samples = [InstructionSample("", "", t) for t in texts]
    return train(samples, cfg, is_pretrain=True, save_dir=save_dir)
