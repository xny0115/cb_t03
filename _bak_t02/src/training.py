from __future__ import annotations
"""Training utilities for the chatbot."""
from pathlib import Path
from typing import Any, Callable, Iterable, Dict
import json
import logging
import time
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, IterableDataset
from .config import Config
from .service.utils import normalize_config, to_config
from .data.loader import QADataset
from .utils.vocab import build_vocab
from .model.transformer import Seq2SeqTransformer
from .tuning.auto import AutoTuner
from .training_cpu import train_cpu, TorchQADataset, collate_fn
from .trainer import EarlyStopper, evaluate
logger = logging.getLogger(__name__)
def _train_epoch(
    loader: DataLoader,
    model: nn.Module,
    crit: nn.Module,
    opt: optim.Optimizer,
    device: str,
    verbose: bool,
    vocab_size: int,
    scaler: torch.cuda.amp.GradScaler | None,
    save_path: Path | None,
    interval: int,
    grad_clip: float,
    amp: bool,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total = 0.0
    for step, (src, tgt) in enumerate(loader, start=1):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(src, tgt[:, :-1])
            loss = crit(out.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
        if scaler:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                if step == 1:
                    logger.info("Grad clip %.1f", grad_clip)
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                if step == 1:
                    logger.info("Grad clip %.1f", grad_clip)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        total += loss.item()
        if verbose:
            logger.debug("step %d loss %.4f", step, loss.item())
        if interval and save_path and step % interval == 0:
            torch.save(model.state_dict(), save_path)
    return total / len(loader)
def train(
    dataset_path: Path,
    cfg: Dict[str, Any] | Config,
    progress_cb: Callable | None = None,
    model_path: Path | None = None,
    start_epoch: int = 0,
    meta_path: Path | None = None,
) -> Path:
    """Train model using dataset and configuration."""
    force_gpu = False
    if isinstance(cfg, Config):
        force_gpu = getattr(cfg, "force_gpu", False)
    else:
        force_gpu = bool(cfg.get("force_gpu", False))
        cfg = to_config(cfg)
    cfg = normalize_config(cfg)
    assert isinstance(cfg.num_epochs, int) and cfg.num_epochs > 0, "epochs must be int"
    ds = QADataset(dataset_path)
    save_path = model_path or Path("models") / "current.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if meta_path:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        if force_gpu:
            raise RuntimeError("GPU required but not available")
        return train_cpu(ds, cfg, save_path, start_epoch, meta_path, progress_cb)
    try:
        torch.cuda.set_per_process_memory_fraction(1.0)
    except Exception:
        pass
    logger.info("Training started...")
    if len(ds) < 50:
        logger.warning("Dataset too small: %d entries", len(ds))
    tuner = AutoTuner(len(ds))
    sugg = tuner.suggest_config()
    params = {
        "batch_size": cfg.batch_size or sugg.batch_size,
        "learning_rate": cfg.learning_rate or sugg.learning_rate,
        "epochs": cfg.num_epochs or sugg.num_epochs,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "save_every": cfg.save_every,
        "gradient_clipping": cfg.gradient_clipping,
        "use_mixed_precision": cfg.use_mixed_precision,
        "weight_decay": cfg.weight_decay,
        "early_stopping": cfg.early_stopping,
        "early_stopping_patience": cfg.early_stopping_patience,
    }
    vocab = build_vocab(ds)
    full_ds = TorchQADataset(ds, vocab)
    val_count = max(1, int(len(full_ds) * 0.1))
    train_count = len(full_ds) - val_count
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_count, val_count]
    )
    persist = params["num_workers"] > 0
    t0 = time.time()
    if isinstance(train_ds, IterableDataset):
        loader = DataLoader(
            train_ds,
            batch_size=params["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=min(2, params["num_workers"]),
            pin_memory=False,
        )
    else:
        loader = DataLoader(
            train_ds,
            batch_size=params["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=params["num_workers"],
            pin_memory=params["pin_memory"],
            persistent_workers=persist,
        )
    logger.info("prefetch ready in %.2fs", time.time() - t0)
    val_loader = DataLoader(
        val_ds,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
        persistent_workers=persist,
    )
    device = params["device"]
    scaler = torch.cuda.amp.GradScaler(enabled=params["use_mixed_precision"] and device == "cuda")
    model = Seq2SeqTransformer(vocab_size=len(vocab))
    if start_epoch and save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    model.to(device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    max_epochs = params["epochs"]
    stopper = EarlyStopper(params["early_stopping_patience"])
    for epoch in range(start_epoch, max_epochs):
        start_t = time.perf_counter()
        epoch_loss = _train_epoch(
            loader,
            model,
            criterion,
            optimizer,
            device,
            cfg.verbose,
            len(vocab),
            scaler,
            save_path if params["save_every"] else None,
            params["save_every"],
            params["gradient_clipping"],
            params["use_mixed_precision"],
        )
        val_loss = evaluate(model, val_loader, device)
        logger.info(
            "epoch %d/%d | loss=%.4f | val=%.4f | %.0fms",
            epoch + 1,
            max_epochs,
            epoch_loss,
            val_loss,
            (time.perf_counter() - start_t) * 1000.0,
        )
        if progress_cb:
            progress_cb(epoch + 1, max_epochs, val_loss)
        if params["early_stopping"] and stopper(val_loss):
            logger.info("early stopping triggered")
            break
        if not params["save_every"]:
            torch.save(model.state_dict(), save_path)
        if meta_path:
            json.dump(
                {
                    "epochs_done": epoch + 1,
                    "update_time": datetime.utcnow().isoformat(),
                },
                open(meta_path, "w"),
            )
    if params["save_every"]:
        torch.save(model.state_dict(), save_path)
    vocab_path = save_path.with_suffix(".vocab.json")
    json.dump(vocab, open(vocab_path, "w"), ensure_ascii=False)
    logger.info("Model saved to models/current.pth; Training complete")
