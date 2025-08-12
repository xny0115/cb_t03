# === LLM SERVICE DIRECTION ===============================
# 이 엔드포인트/모델 서빙은 LLM 기준으로 설계됩니다.
# 추론 로직은 토큰화/컨텍스트 길이/마스크 일관성을 우선 검증합니다.
# =========================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List
import logging
import os
import hashlib
import re
import torch

try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
except ImportError:
    pynvml = None


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

        if pynvml:
            try:
                pynvml.nvmlInit()
                self._pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._pynvml_handle = None
        else:
            self._pynvml_handle = None


        hf_name = os.getenv("HF_MODEL_NAME")
        if hf_name:
            self.model = HFModel(hf_name)
        elif self.model_path.exists():
            try:
                self.model, _ = load_transformer(self.model_path)
                logging.getLogger(__name__).info("[SERVE] model_loaded path=%s params=%s", str(self.model_path), sum(p.numel() for p in self.model.parameters()))
                spm_model_path = str(self._config.get("spm_model_path", "tokenizer/spm.model"))
                if Path(spm_model_path).exists():
                    self.tokenizer = SentencePieceTokenizer(spm_model_path)
                else:
                    logging.getLogger(__name__).warning(f"SPM model not found at {spm_model_path}, tokenizer not loaded.")
            except Exception:
                self.model = load_model(self.model_path)
                logging.getLogger(__name__).info("[SERVE] model_loaded path=%s params=%s", str(self.model_path), sum(p.numel() for p in self.model.parameters()))

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
            logger.warning("CUDA not available: proceeding on CPU (slow).")

        logger.info("training mode=%s, device=%s", mode, device)

        cfg = self._config
        logger.info("[CFG] epochs=%s batch=%s lr=%s dropout=%s amp=%s grad_clip=%s resume=%s", cfg.get('num_epochs'), cfg.get('batch_size'), cfg.get('learning_rate'), cfg.get('dropout_ratio'), cfg.get('use_mixed_precision'), cfg.get('grad_clip'), cfg.get('resume'))
        if mode == "pretrain":
            from ..data.subtitle_cleaner import clean_subtitle_files
            clean_subtitle_files(Path("."), self.pretrain_dir)
            data = load_pretrain_dataset(self.pretrain_dir)
            logger.info("[DATA] pretrain size=%s", (len(data) if hasattr(data, "__len__") else "?"))
            self.model_path = self.model_dir / "pretrain.pth"
            model, tokenizer = train_pretrain(data, cfg, save_dir=str(self.model_path.parent))
        elif mode == "resume":
            is_pretrain_resume = (self.model_dir / "pretrain.pth").exists() or \
                                 (self.model_dir / "last_pretrain.ckpt").exists()

            cfg["resume"] = True

            if is_pretrain_resume:
                data = load_pretrain_dataset(self.pretrain_dir)
                self.model_path = self.model_dir / "pretrain.pth"
                cfg["model_path"] = str(self.model_path)
                model, tokenizer = train_pretrain(data, cfg, save_dir=str(self.model_path.parent))
            else:
                data = load_instruction_dataset(self.finetune_dir)
                self.model_path = self.model_dir / "finetune.pth"
                cfg["model_path"] = str(self.model_path)
                model, tokenizer = train_transformer(data, cfg, save_dir=str(self.model_path.parent))
        else:
            data = load_instruction_dataset(self.finetune_dir)
            logger.info("[DATA] finetune size=%s", (len(data) if hasattr(data, "__len__") else "?"))
            self.model_path = self.model_dir / "finetune.pth"
            model, tokenizer = train_transformer(data, cfg, save_dir=str(self.model_path.parent))

        # Note: save_transformer is now called inside train()

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
        if isinstance(self.model, Seq2SeqTransformer) and not self.tokenizer:
            return {"success": False, "msg": "no_tokenizer", "data": None}
        if not text.strip():
            return {"success": False, "msg": "empty_input", "data": None}
        if len(text) > self.MAX_INPUT_LEN:
            return {"success": False, "msg": "too_long", "data": None}

        if isinstance(self.model, Seq2SeqTransformer) and self.tokenizer:
            ids = self.tokenizer.encode(text, True)
            src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            src = src.to(next(self.model.parameters()).device)
            logging.getLogger(__name__).info("[GEN] max_new_tokens=%s", 50)
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
        data = {}
        if psutil:
            data["cpu_usage"] = psutil.cpu_percent()
        else:
            data["cpu_usage"] = 0.0

        if self._pynvml_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._pynvml_handle)
                data["gpu_usage"] = float(util.gpu)
            except Exception:
                data["gpu_usage"] = None
        else:
            data["gpu_usage"] = None

        return {"success": True, "msg": "idle", "data": data}

    def auto_tune(self) -> Dict[str, Any]:
        """Apply AutoTuner suggestions to config."""
        size, tokens, txt_lines, json_lines, skipped = get_dataset_info(
            self.pretrain_dir, self.finetune_dir, self.data_dir / "additional"
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
