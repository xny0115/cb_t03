from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import logging
import os
import torch

from ..data.loader import load_dataset
from ..model import (
    DummyModel,
    HFModel,
    load_model,
    save_model,
    Seq2SeqTransformer,
    save_transformer,
    load_transformer,
)
from ..training.simple import train as train_transformer
from ..utils.tokenizer import CharTokenizer


class ChatbotService:
    """Instruction 기반 챗봇 서비스."""

    MAX_INPUT_LEN = 1000

    def __init__(self) -> None:
        self.data_dir = Path("datas")
        self.model_path = Path("models/current.pth")
        self.dataset = load_dataset(self.data_dir)
        self.model: DummyModel | HFModel | Seq2SeqTransformer | None = None
        self.tokenizer: CharTokenizer | None = None

        hf_name = os.getenv("HF_MODEL_NAME")
        if hf_name:
            self.model = HFModel(hf_name)
        elif self.model_path.exists():
            try:
                self.model, vocab = load_transformer(self.model_path)
                self.tokenizer = CharTokenizer.from_vocab(vocab)
            except Exception:
                self.model = load_model(self.model_path)

    def start_training(self) -> Dict[str, Any]:
        if isinstance(self.model, HFModel):
            return {"success": True, "msg": "done", "data": None}
        model, tokenizer = train_transformer(self.dataset)
        save_transformer(model, tokenizer.stoi, self.model_path)
        self.model = model
        self.tokenizer = tokenizer
        logger = logging.getLogger(__name__)
        logger.info("Training complete")
        return {"success": True, "msg": "done", "data": None}

    def delete_model(self) -> bool:
        if self.model_path.exists():
            self.model_path.unlink()
            self.model = None
            self.tokenizer = None
            return True
        return False

    def infer(self, text: str) -> Dict[str, Any]:
        if not self.model:
            return {"success": False, "msg": "no_model", "data": None}
        if not text.strip():
            return {"success": False, "msg": "empty_input", "data": None}
        if len(text) > self.MAX_INPUT_LEN:
            return {"success": False, "msg": "too_long", "data": None}
        if isinstance(self.model, Seq2SeqTransformer) and self.tokenizer:
            ids = self.tokenizer.encode(text, True)
            src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            src = src.to(next(self.model.parameters()).device)
            out_ids = self.model.generate(src, max_new_tokens=50, eos_id=self.tokenizer.stoi["<eos>"])
            out_text = self.tokenizer.decode(out_ids.squeeze().tolist()[1:])
            msg = "ok" if out_text else "no_answer"
            return {"success": True, "msg": msg, "data": out_text}
        out = self.model.predict("", text)
        if not out:
            return {"success": True, "msg": "no_answer", "data": ""}
        return {"success": True, "msg": "ok", "data": out}

    def get_status(self) -> Dict[str, Any]:
        return {"success": True, "msg": "idle", "data": {}}
