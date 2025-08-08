from __future__ import annotations

import logging
from typing import Any, Dict

from ..service.service import ChatbotService

logger = logging.getLogger(__name__)

class WebBackend:
    """웹 UI용 API 브리지. 모든 메소드는 UI에 전달할 수 있도록 JSON 직렬화 가능한 dict를 반환해야 합니다."""

    def __init__(self, svc: ChatbotService) -> None:
        self._svc = svc

    def _try_service_call(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Wrapper to catch exceptions from the service layer and return as a dict."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception("An error occurred in the service layer")
            return {"success": False, "msg": str(e), "data": None}

    def start_training(self, mode: str) -> Dict[str, Any]:
        return self._try_service_call(self._svc.start_training, mode)

    def delete_model(self) -> Dict[str, Any]:
        ok = self._svc.delete_model()
        return {"success": ok, "msg": "deleted" if ok else "no_model", "data": None}

    def infer(self, text: str) -> Dict[str, Any]:
        return self._try_service_call(self._svc.infer, text)

    def get_status(self) -> Dict[str, Any]:
        # get_status is expected to be lightweight and non-failing
        return self._svc.get_status()

    def set_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        ok, msg = self._svc.set_config(cfg)
        return {"success": ok, "msg": msg, "data": None}

    def get_config(self) -> Dict[str, Any]:
        return {"success": True, "msg": "ok", "data": self._svc.get_config()}

    def auto_tune(self) -> Dict[str, Any]:
        return self._try_service_call(self._svc.auto_tune)
