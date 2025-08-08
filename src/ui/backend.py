from __future__ import annotations
import logging
from typing import Any, Dict
from ..service.service import ChatbotService

logger = logging.getLogger(__name__)

class WebBackend:
    def __init__(self, svc: ChatbotService) -> None:
        self._svc = svc

    def _try_service_call(self, func, *args, **kwargs) -> Dict[str, Any]:
        try:
            result = func(*args, **kwargs)
            return result if isinstance(result, dict) else {"success": True, "data": result}
        except Exception as e:
            logger.exception("Error in service layer")
            return {"success": False, "msg": str(e)}

    def start_training(self, mode: str) -> Dict[str, Any]:
        return self._try_service_call(self._svc.start_training, mode)
    def delete_model(self) -> Dict[str, Any]:
        return self._try_service_call(self._svc.delete_model)
    def infer(self, text: str) -> Dict[str, Any]:
        return self._try_service_call(self._svc.infer, text)
    def get_config(self) -> Dict[str, Any]:
        cfg = getattr(self._svc, "get_config", lambda: {})()
        return {"success": bool(cfg), "data": cfg}
