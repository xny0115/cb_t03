from __future__ import annotations

from typing import Any, Dict

from ..service.service import ChatbotService


class WebBackend:
    """웹 UI용 API 브리지."""

    def __init__(self, svc: ChatbotService) -> None:
        self._svc = svc

    def start_training(self, mode: str) -> Dict[str, Any]:
        return self._svc.start_training(mode)

    def delete_model(self) -> Dict[str, Any]:
        """STOP 센티넬 파일로 학습 중지를 요청한다."""
        ok = self._svc.delete_model()
        return {"success": ok, "msg": "stop_requested" if ok else "no_model", "data": None}

    def infer(self, text: str) -> Dict[str, Any]:
        return self._svc.infer(text)

    def get_status(self) -> Dict[str, Any]:
        return self._svc.get_status()

    def set_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        ok, msg = self._svc.set_config(cfg)
        return {"success": ok, "msg": msg, "data": None}

    def get_config(self) -> Dict[str, Any]:  # pragma: no cover
        return {"success": True, "msg": "ok", "data": self._svc.get_config()}

    def auto_tune(self) -> Dict[str, Any]:  # pragma: no cover
        cfg = self._svc.auto_tune()
        return {"success": True, "msg": "ok", "data": cfg}
