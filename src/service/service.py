from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List
import logging
import os
import hashlib
import re
import torch

from ..data.loader import (
    load_instruction_dataset,
    load_pretrain_dataset,
    get_dataset_info,
)
from ..config import load_config, save_config
from ..model import (
    DummyModel,
    HFModel,
    load_model,
    Seq2SeqTransformer,
    save_transformer,
    load_transformer,
)
from ..training.simple import train as train_transformer, pretrain as train_pretrain
from ..utils.tokenizer import SentencePieceTokenizer
from ..utils.validator import validate_config
from ..tuning.auto import AutoTuner


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_epoch_metrics(fp: Path) -> List[str]:
    if not fp.exists():
        return []
    lines = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        if "Epoch" in line and "Loss" in line:
            lines.append(line.strip())
    return lines


class ChatbotService:
    """Instruction 기반 챗봇 서비스."""

    MAX_INPUT_LEN = 1000

    def __init__(self) -> None:
        self.data_dir = Path("datas")
        self.pretrain_dir = self.data_dir / "pretrain"
        self.finetune_dir = self.data_dir / "finetune"
        self.model_dir = Path("models")
        self.model_path = self.model_dir / "finetune.pth"
        self.dataset = load_instruction_dataset(self.finetune_dir)
        self.model: DummyModel | HFModel | Seq2SeqTransformer | None = None
        self.tokenizer: SentencePieceTokenizer | None = None
        self._config = load_config()

        hf_name = os.getenv("HF_MODEL_NAME")
        if hf_name:
            self.model = HFModel(hf_name)
        elif self.model_path.exists():
            try:
                self.model, _ = load_transformer(self.model_path)
                spm_model_path = str(self._config.get("spm_model_path", "tokenizer/spm.model"))
                if Path(spm_model_path).exists():
                    self.tokenizer = SentencePieceTokenizer(spm_model_path)
                else:
                    logging.getLogger(__name__).warning(f"SPM model not found at {spm_model_path}, tokenizer not loaded.")
            except Exception:
                self.model = load_model(self.model_path)

    def start_training(self, mode: str) -> Dict[str, Any]:
        """학습 유형에 따라 분기 처리."""
        valid, msg = validate_config(self._config)
        if not valid:
            return {"success": False, "msg": msg, "data": None}
        if isinstance(self.model, HFModel):
            return {"success": True, "msg": "done", "data": None}

        logger = logging.getLogger(__name__)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("CUDA is required for training.")
            return {"success": False, "msg": "CUDA is required for training.", "data": None}

        logger.info("training mode=%s, device=%s", mode, device)
        log_dir = Path("logs")
        prev_files = sorted(log_dir.glob("*.log"))
        prev_metrics = _parse_epoch_metrics(prev_files[-1]) if prev_files else []
        old_size = old_hash = None
        if mode == "pretrain" and self.model_path.exists():
            old_size = self.model_path.stat().st_size
            old_hash = _sha256(self.model_path)

        cfg = self._config
        if mode == "pretrain":
            from ..data.subtitle_cleaner import clean_subtitle_files
            clean_subtitle_files(Path("."), self.pretrain_dir)
            data = load_pretrain_dataset(self.pretrain_dir)
            self.model_path = self.model_dir / "pretrain.pth"
            model, tokenizer = train_pretrain(data, cfg, save_dir=str(self.model_path.parent))
        elif mode == "resume":
            if (self.model_dir / "pretrain.pth").exists():
                data = load_pretrain_dataset(self.pretrain_dir)
                self.model_path = self.model_dir / "pretrain.pth"
            else:
                data = load_instruction_dataset(self.finetune_dir)
                self.model_path = self.model_dir / "finetune.pth"
            cfg["resume"] = True
            cfg["model_path"] = str(self.model_path)
            model, tokenizer = train_transformer(data, cfg, save_dir=str(self.model_path.parent))
        else:
            data = load_instruction_dataset(self.finetune_dir)
            self.model_path = self.model_dir / "finetune.pth"
            model, tokenizer = train_transformer(data, cfg, save_dir=str(self.model_path.parent))

        save_transformer(model, {}, self.model_path)

        if mode == "pretrain" and old_size is not None and old_hash is not None:
            new_size = self.model_path.stat().st_size
            new_hash = _sha256(self.model_path)
            logger.debug("old: %.2fMB / hash: %s", old_size / 1_048_576, old_hash[:6])
            logger.debug("new: %.2fMB / hash: %s", new_size / 1_048_576, new_hash[:6])
            if old_size == new_size and old_hash == new_hash:
                logger.warning("\u26a0 저장 내용 무변화 → 학습 미반영 가능성")

        new_files = sorted(log_dir.glob("*.log"))
        new_metrics = _parse_epoch_metrics(new_files[-1]) if new_files else []
        if prev_metrics and new_metrics == prev_metrics:
            logger.warning("training log duplicated from previous run")

        self.model = model
        self.tokenizer = tokenizer
        logger.info("Training complete")
        return {"success": True, "msg": "done", "data": None}

    def set_config(self, cfg: Dict[str, Any]) -> Tuple[bool, str]:
        valid, msg = validate_config(cfg)
        if not valid:
            return False, msg
        try:
            self._config.update(cfg)
            save_config(self._config)
            return True, "saved"
        except Exception as exc:
            logging.getLogger(__name__).warning("config save failed: %s", exc)
            return False, str(exc)

    def get_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def delete_model(self) -> bool:
        deleted = False
        for fp in self.model_dir.glob("*.pth"):
            try:
                fp.unlink()
                deleted = True
            except FileNotFoundError:
                continue
            except Exception as exc:
                logging.getLogger(__name__).warning("model delete failed: %s", exc)
        if deleted:
            self.model = None
            self.tokenizer = None
        return deleted

    def infer(self, text: str) -> Dict[str, Any]:
        if not self.model:
            return {"success": False, "msg": "no_model", "data": None}
        if not self.tokenizer:
            return {"success": False, "msg": "no_tokenizer", "data": None}
        if not text.strip():
            return {"success": False, "msg": "empty_input", "data": None}
        if len(text) > self.MAX_INPUT_LEN:
            return {"success": False, "msg": "too_long", "data": None}

        if isinstance(self.model, Seq2SeqTransformer) and self.tokenizer:
            ids = self.tokenizer.encode(text, True)
            src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            src = src.to(next(self.model.parameters()).device)
            out_ids = self.model.generate(
                src,
                max_new_tokens=50,
                eos_id=self.tokenizer.eos_id,
                pad_id=self.tokenizer.pad_id,
            )
            out_text = self.tokenizer.decode(out_ids.squeeze().tolist())
            msg = "ok" if out_text else "no_answer"
            return {"success": True, "msg": msg, "data": out_text}

        out = self.model.predict("", text)
        if not out:
            return {"success": True, "msg": "no_answer", "data": ""}
        return {"success": True, "msg": "ok", "data": out}

    def get_status(self) -> Dict[str, Any]:
        return {"success": True, "msg": "idle", "data": {}}

    def auto_tune(self) -> Dict[str, Any]:
        """Apply AutoTuner suggestions to config."""
        size, tokens, txt_lines, json_lines, skipped = get_dataset_info(
            self.pretrain_dir, self.finetune_dir
        )
        print(f"[DEBUG] Found pretrain txt files: {txt_lines} lines")
        print(f"[DEBUG] Found finetune jsonl files: {json_lines} lines")
        if skipped:
            print(f"[DEBUG] Skipped files: {skipped}")
        print(f"[DEBUG] AutoTune triggered: dataset size = {size}, tokens = {tokens}")
        cfg = AutoTuner(size, tokens).suggest()
        valid, msg = validate_config(cfg)
        if not valid:
            return {"success": False, "msg": msg, "data": None}
        self._config.update(cfg)
        save_config(self._config)
        logging.getLogger(__name__).info("auto-tune applied: %s", cfg)
        return {"success": True, "msg": "auto-tune applied", "data": cfg}
