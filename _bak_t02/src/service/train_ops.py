from __future__ import annotations

"""Helper operations for :class:`ChatbotService`."""

from pathlib import Path
from threading import Thread
from typing import Any, Dict
import json
import logging
import time
import torch

from .utils import to_config, normalize_config, _CFG_MAP
from ..training import train
from ..utils.logger import log_gpu_memory, LOG_PATH
from .loader import load_model
from .rewriter import rewrite as _rewrite

logger = logging.getLogger(__name__)


def run_training(service: "ChatbotService", path: Path) -> None:
    """Synchronous training helper."""
    cfg = service._config
    epochs = int(cfg.get("epochs", 20))
    logger.info("training epochs=%d", epochs)
    last = 0.0

    def progress(epoch: int, total: int, loss: float) -> None:
        nonlocal last
        with service._lock:
            service._status_msg = f"{epoch}/{total} loss={loss:.4f}"
            service._progress = epoch / total
            service._last_loss = loss
        if time.time() - last >= 1:
            last = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device selected: %s", device)
    log_gpu_memory()
    meta_path = service.model_path.with_suffix(".meta.json")
    start_ep = 0
    cfg_local = cfg.copy()
    if service.model_path.exists() and meta_path.exists():
        try:
            meta = json.load(open(meta_path))
            start_ep = int(meta.get("epochs_done", 0))
        except Exception:
            start_ep = 0
        logger.info("resume training from epoch %d", start_ep)
        if start_ep >= epochs:
            logger.info("continue training for %d more epochs", epochs)
            cfg_local["epochs"] = start_ep + epochs
    elif meta_path.exists() and not service.model_path.exists():
        try:
            meta_path.unlink()
        except Exception:
            pass
    try:
        norm = normalize_config(to_config(cfg_local)).__dict__
        cfg_local.update({src: norm[key] for key, (src, _) in _CFG_MAP.items()})
        train(
            path,
            cfg_local,
            progress_cb=progress,
            model_path=service.model_path,
            start_epoch=start_ep,
            meta_path=meta_path,
        )
        if (
            not service.model_path.exists()
            or service.model_path.stat().st_size < 1_000_000
        ):
            raise RuntimeError("모델 저장 실패: 생성 실패 또는 용량 미달")
        msg = "done"
        service.model_exists = True
        logger.info("Training complete")
        service._tokenizer, service._model = load_model(service.model_path)
    except Exception as exc:  # pragma: no cover - best effort
        logger.exception("Training failed")
        with service._lock:
            service._status_msg = f"error: {exc}"
            service._progress = 0.0
            service.training = False
        service.model_exists = False
        return
    with service._lock:
        service.training = False
        service._status_msg = msg
        service._progress = 1.0 if msg == "done" else 0.0


def start_training(service: "ChatbotService", data_path: str = ".") -> Dict[str, Any]:
    """Asynchronous training helper."""
    logging.getLogger().setLevel(logging.DEBUG)
    with service._lock:
        if service.training:
            if service._thread is None or not service._thread.is_alive():
                logger.debug("stale training flag detected; resetting")
                service.training = False
            else:
                logger.warning("Training already running")
                return {"success": False, "msg": "already_training", "data": None}
        service.training = True
        service._status_msg = "starting"
        service._progress = 0.0

    meta_path = service.model_path.with_suffix(".meta.json")
    start_ep = 0
    cfg_local = service._config.copy()
    if service.model_path.exists() and meta_path.exists():
        try:
            meta = json.load(open(meta_path))
            start_ep = int(meta.get("epochs_done", 0))
        except Exception:
            start_ep = 0
        logger.info("resume training from epoch %d", start_ep)
        epochs = int(cfg_local.get("epochs", 20))
        if start_ep >= epochs:
            logger.info("continue training for %d more epochs", epochs)
            cfg_local["epochs"] = start_ep + epochs
    elif meta_path.exists() and not service.model_path.exists():
        try:
            meta_path.unlink()
        except Exception:
            pass

    def progress(epoch: int, total: int, loss: float) -> None:
        with service._lock:
            service._status_msg = f"{epoch}/{total} loss={loss:.4f}"
            service._progress = epoch / total
            service._last_loss = loss

    def runner() -> None:
        try:
            norm = normalize_config(to_config(cfg_local)).__dict__
            cfg_n = cfg_local.copy()
            cfg_n.update({src: norm[key] for key, (src, _) in _CFG_MAP.items()})
            train(
                Path("datas") / data_path,
                cfg_n,
                progress_cb=progress,
                model_path=service.model_path,
                start_epoch=start_ep,
                meta_path=meta_path,
            )
            if (
                not service.model_path.exists()
                or service.model_path.stat().st_size < 1_000_000
            ):
                raise RuntimeError("모델 저장 실패: 생성 실패 또는 용량 미달")
            logger.info("Model saved to %s", service.model_path)
            msg = "done"
            service.model_exists = True
            logger.info("Training complete")
            service._tokenizer, service._model = load_model(service.model_path)
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("Training failed")
            msg = f"error: {exc}"
            service.model_exists = False
        with service._lock:
            service.training = False
            service._status_msg = msg
            service._progress = 1.0 if msg == "done" else 0.0

    log_gpu_memory()
    service._thread = Thread(target=runner, daemon=True)
    service._thread.start()
    return {"success": True, "msg": "started", "data": None}
