from __future__ import annotations

"""Inference and status helper functions for ChatbotService."""

from typing import Any, Dict
import logging
from pathlib import Path
import torch

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None
try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional
    pynvml = None

from .rewriter import rewrite as _rewrite
from ..generate_utils import generate_response
from ..utils.logger import LOG_PATH

logger = logging.getLogger(__name__)


def infer_text(service: "ChatbotService", text: str) -> Dict[str, Any]:
    """Inference helper."""
    logger.info("infer | model_exists=%s", service.model_exists)
    if not service.model_exists or service._model is None or service._tokenizer is None:
        return {"success": False, "msg": "model_missing", "data": None}
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda" and not service._config.get("force_cpu", False):
            try:
                assert torch.cuda.current_device() >= 0
            except Exception:
                device = torch.device("cpu")
        logger.info("Device selected: %s", device)
        if hasattr(service._model, "to"):
            service._model.to(device).eval()
        answer = generate_response(service._model, service._tokenizer, text, service._config)
        words = [w for w in answer.split() if not w.startswith("[")]
        tokens_l = []
        known_pos = True
        for w in words:
            pos = service._pos_map.get(w)
            if pos is None:
                known_pos = False
                pos = "NNG"
            tokens_l.append({"lemma": w, "pos": pos})
        if known_pos:
            try:
                from ..utils.restorer import restore_sentence
                answer = restore_sentence(tokens_l)
            except Exception:
                pass
        logger.debug("infer result: %s", answer[:60])
        return {"success": True, "msg": "", "data": answer}
    except Exception as exc:  # pragma: no cover - best effort
        logger.exception("infer failed")
        return {"success": False, "msg": f"error: {exc}", "data": None}


def rewrite_text(service: "ChatbotService", text: str) -> Dict[str, Any]:
    """Dataset based paraphrasing."""
    try:
        out = _rewrite(text)
        return {"success": True, "msg": "", "data": out}
    except Exception as exc:  # pragma: no cover - best effort
        logger.exception("rewrite failed")
        return {"success": False, "msg": f"error: {exc}", "data": None}


def get_status(service: "ChatbotService") -> Dict[str, Any]:
    """Return current status dict."""
    with service._lock:
        msg = service._status_msg
        running = service.training
        progress = service._progress
        last_loss = service._last_loss
    data = {
        "training": running,
        "progress": progress,
        "status_msg": msg,
        "last_loss": last_loss,
        "model_exists": service.model_exists,
    }
    if psutil is not None:
        data["cpu_usage"] = psutil.cpu_percent()
    else:
        data["cpu_usage"] = 0.0
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            data["gpu_usage"] = float(util.gpu)
        except Exception:
            data["gpu_usage"] = None
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    else:
        data["gpu_usage"] = None
    try:
        lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
        data["logs"] = "\n".join(lines[-2000:])
    except Exception:
        data["logs"] = ""
    data["current_config"] = service._config.copy()
    return {"success": True, "msg": "ok", "data": data}


def delete_model(service: "ChatbotService") -> bool:
    """Delete existing model file if present."""
    if service.training:
        service.stop_training()
    if service.model_path.exists():
        try:
            service.model_path.unlink()
            meta = service.model_path.with_suffix(".meta.json")
            if meta.exists():
                meta.unlink()
            service.model_exists = False
            service._status_msg = "model_deleted"
            logger.info("Model deleted")
            return True
        except Exception:  # pragma: no cover - best effort
            logger.exception("Delete model failed")
            return False
    return False
