from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..data.loader import load_dataset
from ..model.dummy import DummyModel, load_model, save_model


class ChatbotService:
    """Instruction 기반 챗봇 서비스."""

    def __init__(self) -> None:
        self.data_dir = Path("datas")
        self.model_path = Path("models/current.pth")
        self.dataset = load_dataset(self.data_dir)
        self.model: DummyModel | None = None
        if self.model_path.exists():
            self.model = load_model(self.model_path)

    def start_training(self) -> Dict[str, Any]:
        mapping = {f"{s.instruction} {s.input}".strip(): s.output for s in self.dataset}
        self.model = DummyModel(mapping)
        save_model(self.model, self.model_path)
        return {"success": True, "msg": "done", "data": None}

    def delete_model(self) -> bool:
        if self.model_path.exists():
            self.model_path.unlink()
            self.model = None
            return True
        return False

    def infer(self, text: str) -> Dict[str, Any]:
        if not self.model:
            return {"success": False, "msg": "no_model", "data": None}
        out = self.model.predict("", text)
        return {"success": True, "msg": "ok", "data": out}

    def get_status(self) -> Dict[str, Any]:
        return {"success": True, "msg": "idle", "data": {}}
