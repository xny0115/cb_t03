from __future__ import annotations
from typing import List, Any, Dict, Tuple, Optional
import logging, time
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from ..data.loader import InstructionSample
from ..model.transformer import Seq2SeqTransformer
from ..utils.tokenizer import SentencePieceTokenizer
from .helpers import PairDataset, timed_collate

logger = logging.getLogger(__name__)

def _prepare_dataset(samples, tokenizer, is_pretrain):
    pairs = []
    for s in samples:
        if is_pretrain:
            encoded = tokenizer.encode(s.output, add_special_tokens=True)
            src, tgt = encoded, encoded
        else:
            src = tokenizer.encode(f"{s.instruction} {s.input}".strip(), add_special_tokens=True)
            tgt = tokenizer.encode(s.output, add_special_tokens=True)
        pairs.append((src, tgt))
    return PairDataset(pairs), len(samples)

def _create_loader(dataset, cfg):
    """Return DataLoader with device aware pin_memory."""
    pin = bool(cfg.get("pin_memory", torch.cuda.is_available()))
    workers = int(cfg.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=True,
        collate_fn=timed_collate,
        num_workers=workers,
        pin_memory=pin,
        drop_last=True,
    )

def _init_model(tokenizer, cfg):
    return Seq2SeqTransformer(vocab_size=tokenizer.vocab_size, embed_dim=int(cfg.get("model_dim", 256)), num_heads=int(cfg.get("num_heads", 8)), num_encoder_layers=int(cfg.get("num_encoder_layers", 6)), num_decoder_layers=int(cfg.get("num_decoder_layers", 6)), dim_ff=int(cfg.get("ff_dim", 1024)), dropout=float(cfg.get("dropout_ratio", 0.1)))

def train(samples, cfg, *, is_pretrain=False, model=None):
    tokenizer = SentencePieceTokenizer(Path(cfg.get("tokenizer_path", "models/spm_bpe_8k.model")))
    epochs = int(cfg.get("num_epochs", 5))
    dataset, line_count = _prepare_dataset(samples, tokenizer, is_pretrain)
    train_size = int(0.95 * len(dataset))
    if train_size < 1: raise ValueError("Dataset too small.")
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    loader = _create_loader(train_set, cfg)
    val_loader = _create_loader(val_set, cfg)
    if model is None:
        logger.info("Initializing a new model.")
        model = _init_model(tokenizer, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    amp = cuda
    logger.info(
        f"CUDA={cuda} AMP={amp} device_name={torch.cuda.get_device_name() if cuda else 'cpu'}"
    )
    model.to(device)
    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    opt = optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(loader) * epochs)
    best_val_loss = float("inf")
    checkpoint_path = Path("models/training_state.pth")
    start_epoch = 0
    global_step = 0
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])  # type: ignore[arg-type]
        opt.load_state_dict(ckpt["optimizer"])  # type: ignore[arg-type]
        scheduler.load_state_dict(ckpt["scheduler"])  # type: ignore[arg-type]
        if amp and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])  # type: ignore[arg-type]
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_metric", float("inf")))
        logger.info(
            f"Resumed from epoch {start_epoch-1} step {global_step} best {best_val_loss:.4f}"
        )
    for epoch in range(start_epoch, epochs):
        train_loss, _, global_step, _ = _train_epoch(
            loader,
            model,
            crit,
            opt,
            scaler,
            scheduler,
            tokenizer,
            device,
            amp,
            global_step,
        )
        val_loss = _eval_epoch(val_loader, model, crit, tokenizer, device, amp)
        logger.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_val_loss,
            "cfg": cfg,
        }
        torch.save(state, checkpoint_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    logger.info("Training complete")
    return model

def _train_epoch(
    loader,
    model,
    crit,
    opt,
    scaler,
    scheduler,
    tokenizer,
    device,
    amp=True,
    global_step=0,
):
    model.train()
    total_loss = 0
    step = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        if tgt.size(1) < 2:
            continue
        try:
            opt.zero_grad(set_to_none=True)  # type: ignore[arg-type]
        except TypeError:
            opt.zero_grad()
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(src, tgt_in, pad_id=tokenizer.pad_id)
            loss = crit(logits.view(-1, tokenizer.vocab_size), tgt_out.reshape(-1))
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            step += 1
            global_step += 1
    avg = total_loss / step if step > 0 else 0
    last_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0.0
    return avg, step, global_step, last_lr

def _eval_epoch(loader, model, crit, tokenizer, device, amp):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            if tgt.size(1) < 2:
                continue
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(src, tgt_in, pad_id=tokenizer.pad_id)
                loss = crit(logits.view(-1, tokenizer.vocab_size), tgt_out.reshape(-1))
            if torch.isfinite(loss):
                total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0

def pretrain(texts, cfg=None, model=None):
    return train([InstructionSample("", "", t) for t in texts], cfg or {}, is_pretrain=True, model=model)
