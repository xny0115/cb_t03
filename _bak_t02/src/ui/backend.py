from __future__ import annotations

"""pywebview API bridge."""

from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

from ..service import ChatbotService


class WebBackend:
    """Expose limited methods for the web UI."""

    __slots__ = ("_svc",)

    def __init__(self, svc: ChatbotService) -> None:
        self._svc = svc

    def start_training(self) -> Dict[str, Any]:
        return self._svc.start_training()

    def auto_tune(self) -> Dict[str, Any]:
        cfg = self._svc.auto_tune()
        return cfg

    def get_config(self) -> Dict[str, Any]:
        return self._svc.get_config()

    def set_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        ok, msg = self._svc.update_config(cfg)
        return {"success": ok, "msg": msg, "data": None}

    def delete_model(self) -> Dict[str, Any]:
        ok = self._svc.delete_model()
        return {"success": ok, "msg": "deleted" if ok else "no_model", "data": None}

    def infer(self, text: str) -> Dict[str, Any]:
        logger.debug("API infer called: %s", text[:40])
        return self._svc.infer(text)

    def rewrite(self, text: str) -> Dict[str, Any]:
        logger.debug("API rewrite called: %s", text[:40])
        return self._svc.rewrite(text)

    def get_status(self) -> Dict[str, Any]:
        return self._svc.get_status()
