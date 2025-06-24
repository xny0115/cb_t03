from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os

from ..data.loader import load_dataset
from ..model import DummyModel, HFModel, load_model, save_model


class ChatbotService:
    """Instruction 기반 챗봇 서비스."""

    MAX_INPUT_LEN = 1000

    def __init__(self) -> None:
        self.data_dir = Path("datas")
        self.model_path = Path("models/current.pth")
        self.dataset = load_dataset(self.data_dir)
        self.model: DummyModel | HFModel | None = None

        hf_name = os.getenv("HF_MODEL_NAME")
        if hf_name:
            self.model = HFModel(hf_name)
        elif self.model_path.exists():
            self.model = load_model(self.model_path)

    def start_training(self) -> Dict[str, Any]:
        if isinstance(self.model, HFModel):
            return {"success": True, "msg": "done", "data": None}
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
        if not text.strip():
            return {"success": False, "msg": "empty_input", "data": None}
        if len(text) > self.MAX_INPUT_LEN:
            return {"success": False, "msg": "too_long", "data": None}
        out = self.model.predict("", text)
        if not out:
            return {"success": True, "msg": "no_answer", "data": ""}
        return {"success": True, "msg": "ok", "data": out}

    def get_status(self) -> Dict[str, Any]:
        return {"success": True, "msg": "idle", "data": {}}
