from __future__ import annotations
# pragma: no cover

"""FastAPI and local backend entrypoints."""  # pragma: no cover

from fastapi import FastAPI  # pragma: no cover
from typing import Any, Dict

from .service import ChatbotService
from .utils.logger import setup_logger

setup_logger()
service = ChatbotService()
app = FastAPI()


@app.get("/config")
def get_config() -> Dict[str, Any]:
    cfg = service.get_config()
    return {"success": True, "data": cfg.__dict__, "error": None}


@app.post("/config")
def set_config(conf: Dict[str, Any]) -> Dict[str, Any]:
    cfg = service.set_config(conf)
    return {"success": True, "data": cfg.__dict__, "error": None}


@app.post("/api/set_config")
def api_set_config(conf: Dict[str, Any]) -> Dict[str, Any]:
    """Backward compatible config update endpoint."""
    return set_config(conf)


@app.post("/train")
def start_train() -> Dict[str, Any]:
    service.start_training(".")
    return {"success": True, "data": {"message": "training started"}, "error": None}


@app.post("/infer")
def do_infer(payload: Dict[str, str]) -> Dict[str, Any]:
    answer = service.infer(payload.get("question", ""))
    return {"success": True, "data": {"answer": answer}, "error": None}


@app.get("/status")
def status() -> Dict[str, Any]:
    return {"success": True, "data": service.get_status(), "error": None}


class Backend:
    """Used by tests and webview."""

    def __init__(self) -> None:
        self.svc = service

    def get_config(self) -> Dict[str, Any]:
        return get_config()

    def update_config(self, conf: Dict[str, Any]) -> Dict[str, Any]:
        return set_config(conf)

    def start_train(self, data_path: str = ".") -> Dict[str, Any]:
        try:
            self.svc.start_training(data_path)
            return {"success": True, "data": {"message": "training started"}, "error": None}
        except Exception as exc:
            return {"success": False, "data": None, "error": str(exc)}

    def get_status(self) -> Dict[str, Any]:
        return status()

    def inference(self, question: str) -> Dict[str, Any]:
        try:
            ans = self.svc.infer(question)
            return {"success": True, "data": {"answer": ans}, "error": None}
        except Exception as exc:
            return {"success": False, "data": None, "error": str(exc)}

    def delete_model(self) -> bool:
        return self.svc.delete_model()
