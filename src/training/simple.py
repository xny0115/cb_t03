from __future__ import annotations
from typing import List, Any, Dict, Tuple, Optional
import logging
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from ..data.loader import InstructionSample
from ..model.transformer import Seq2SeqTransformer
from ..utils.tokenizer import SentencePieceTokenizer
from .helpers import PairDataset, timed_collate

logger = logging.getLogger(__name__)
def _sync_vocab_size(model: Seq2SeqTransformer, vocab: int) -> None:
    if model.embed.num_embeddings == vocab and model.fc_out.out_features == vocab:
        return
    old_embed, old_out = model.embed, model.fc_out
    model.embed = nn.Embedding(vocab, old_embed.embedding_dim)
    model.fc_out = nn.Linear(old_out.in_features, vocab)
    with torch.no_grad():
        n = min(old_embed.num_embeddings, vocab)
        model.embed.weight[:n] = old_embed.weight[:n]
        n = min(old_out.out_features, vocab)
        model.fc_out.weight[:n] = old_out.weight[:n]
        if old_out.bias is not None and model.fc_out.bias is not None:
            model.fc_out.bias[:n] = old_out.bias[:n]


def _dry_run(loader: DataLoader, model: Seq2SeqTransformer, crit: nn.Module, tokenizer: SentencePieceTokenizer, device: torch.device, amp: bool) -> None:
    try:
        src, tgt = next(iter(loader))
        src, tgt = src.to(device), tgt.to(device)
        if tgt.size(1) < 2:
            raise RuntimeError("Batch too small")
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with torch.cuda.amp.autocast(enabled=amp):
            loss = crit(model(src, tgt_in, pad_id=tokenizer.pad_id).view(-1, tokenizer.vocab_size), tgt_out.reshape(-1))
        loss.backward(); model.zero_grad(); logger.info("dry_run=success")
    except Exception as e:  # pragma: no cover
        logger.error("dry_run=fail %s", e); raise

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
    import platform
    workers = int(cfg.get("num_workers", 0))
    pin = bool(cfg.get("pin_memory", False))
    if platform.system() == "Windows":
        workers = 0
        pin = False
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=True,
        collate_fn=timed_collate,
        num_workers=workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=False,
    )

def _init_model(tokenizer, cfg):
    return Seq2SeqTransformer(vocab_size=tokenizer.vocab_size, embed_dim=int(cfg.get("model_dim", 256)), num_heads=int(cfg.get("num_heads", 8)), num_encoder_layers=int(cfg.get("num_encoder_layers", 6)), num_decoder_layers=int(cfg.get("num_decoder_layers", 6)), dim_ff=int(cfg.get("ff_dim", 1024)), dropout=float(cfg.get("dropout_ratio", 0.1)))

def train(
    samples: List[InstructionSample],
    cfg: Dict[str, Any],
    *,
    is_pretrain: bool = False,
    model: Optional[Seq2SeqTransformer] = None,
    resume: bool = False,
) -> Seq2SeqTransformer:
    tok_path = Path(cfg.get("tokenizer_path", "models/spm_bpe_8k.model"))
    tokenizer = SentencePieceTokenizer(tok_path)
    logger.info("tokenizer_path=%s vocab_size=%d", tok_path, tokenizer.vocab_size)
    logger.info(
        "train(): epochs=%s max_steps=%s amp=%s",
        cfg.get("num_epochs"),
        cfg.get("max_steps", 0),
        cfg.get("use_mixed_precision"),
    )
    dataset, line_count = _prepare_dataset(samples, tokenizer, is_pretrain)
    if line_count <= 0:
        raise ValueError("No training samples found")
    logger.info("train_samples=%d", line_count)
    epochs = int(cfg.get("num_epochs", 0))
    max_steps = int(cfg.get("max_steps", 0))
    if epochs <= 0 and max_steps <= 0:
        raise ValueError("max_steps or num_epochs must be > 0")
    train_size = int(0.95 * len(dataset))
    if train_size < 1:
        raise ValueError("Dataset too small.")
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    loader = _create_loader(train_set, cfg)
    val_loader = _create_loader(val_set, cfg)
    ckpt_path = Path(cfg.get("checkpoint_path", "models/training_state.pth"))
    resume = bool(cfg.get("resume", resume))
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if resume and ckpt_path.exists():
        logger.info("Resume training from checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg_ckpt = ckpt.get("cfg", cfg)
        model = _init_model(tokenizer, cfg_ckpt)
        model.load_state_dict(ckpt["model"])  # type: ignore[arg-type]
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_metric", float("inf")))
        mode = "RESUME"
    else:
        if resume:
            logger.warning(
                "Resume is true but checkpoint not found at %s. Starting fresh.",
                ckpt_path,
            )
        if model is None:
            logger.info("Initializing a new model for cold start.")
            model = _init_model(tokenizer, cfg)
        mode = "COLD_START"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    logger.info("CUDA=%s device_name=%s", cuda, torch.cuda.get_device_name() if cuda else "cpu")
    model.to(device)
    _sync_vocab_size(model, tokenizer.vocab_size)
    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    opt = optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    amp = cuda
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, len(loader)) * max(1, epochs)
    )
    if resume:
        opt.load_state_dict(ckpt["optimizer"])  # type: ignore[arg-type]
        scheduler.load_state_dict(ckpt["scheduler"])  # type: ignore[arg-type]
        if amp and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])  # type: ignore[arg-type]
        logger.info(
            "Resumed from epoch %d step %d best %.4f",
            start_epoch - 1,
            global_step,
            best_val_loss,
        )
    logger.info("MODE=%s, ckpt_path=%s, vocab_size=%d", mode, ckpt_path, tokenizer.vocab_size)
    try:
        _dry_run(
            loader,
            model,
            crit,
            tokenizer,
            device,
            bool(cfg.get("use_mixed_precision", False)),
        )
    except Exception as e:
        logger.error("dry_run_failed: %s", e)
        raise
    for epoch in range(start_epoch, max(epochs, 1)):
        train_loss, steps, global_step, _ = _train_epoch(
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
            max_steps,
        )
        val_loss = _eval_epoch(val_loader, model, crit, tokenizer, device, amp)
        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f",
            epoch + 1,
            max(epochs, 1),
            train_loss,
            val_loss,
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
        torch.save(state, ckpt_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if max_steps and global_step >= max_steps:
            break
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
    max_steps=0,
):
    model.train()
    total_loss = 0
    step = 0
    for src, tgt in loader:
        if max_steps and global_step >= max_steps:
            break
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

def pretrain(texts, cfg=None, model=None, resume: bool = False):
    return train(
        [InstructionSample("", "", t) for t in texts],
        cfg or {},
        is_pretrain=True,
        model=model,
        resume=resume,
    )
