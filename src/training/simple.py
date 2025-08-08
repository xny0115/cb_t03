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
    return DataLoader(dataset, batch_size=int(cfg.get("batch_size", 32)), shuffle=True, collate_fn=timed_collate, num_workers=int(cfg.get("num_workers", 0)), pin_memory=True, drop_last=True)

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
    if device.type == 'cpu': raise RuntimeError("CUDA GPU not available.")
    logger.info(f"Training on device: {device}")
    model.to(device)
    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    opt = optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("use_mixed_precision", True)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(loader) * epochs)
    best_val_loss = float('inf')
    checkpoint_path = Path("models/best_model_checkpoint.pth")
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        best_val_loss = _eval_epoch(val_loader, model, crit, tokenizer, device)
        logger.info(f"Resumed from checkpoint. Initial validation loss: {best_val_loss:.4f}")
    for epoch in range(epochs):
        train_loss = _train_epoch(loader, model, crit, opt, scaler, scheduler, tokenizer, device)
        val_loss = _eval_epoch(val_loader, model, crit, tokenizer, device)
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"New best model saved to {checkpoint_path}")
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path))
    return model

def _train_epoch(loader, model, crit, opt, scaler, scheduler, tokenizer, device):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        if tgt.size(1) < 2: continue
        opt.zero_grad(set_to_none=True)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(src, tgt_in, pad_id=tokenizer.pad_id)
            loss = crit(logits.view(-1, tokenizer.vocab_size), tgt_out.reshape(-1))
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0

def _eval_epoch(loader, model, crit, tokenizer, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            if tgt.size(1) < 2: continue
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(src, tgt_in, pad_id=tokenizer.pad_id)
                loss = crit(logits.view(-1, tokenizer.vocab_size), tgt_out.reshape(-1))
            if torch.isfinite(loss):
                total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0

def pretrain(texts, cfg=None, model=None):
    return train([InstructionSample("", "", t) for t in texts], cfg or {}, is_pretrain=True, model=model)
