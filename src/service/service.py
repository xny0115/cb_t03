from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List
import logging
import os
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
    Seq2SeqTransformer,
    save_transformer,
    load_transformer,
)
from ..training.simple import train as train_transformer, pretrain
from ..utils.tokenizer import SentencePieceTokenizer
from ..utils.validator import validate_config
from ..tuning.auto import AutoTuner

logger = logging.getLogger(__name__)

class ChatbotService:
    '''Instruction 기반 챗봇 서비스.'''

    MAX_INPUT_LEN = 1000

    def __init__(self) -> None:
        self.data_dir = Path("datas")
        self.pretrain_dir = self.data_dir / "pretrain"
        self.finetune_dir = self.data_dir / "finetune"
        self.additional_dir = self.data_dir / "additional_finetune"
        self.model_dir = Path("models")
        
        self.model_paths = {
            "pretrain": self.model_dir / "pretrain.pth",
            "finetune": self.model_dir / "finetune.pth",
            "additional_finetune": self.model_dir / "additional_finetune.pth",
        }
        
        self.model: DummyModel | HFModel | Seq2SeqTransformer | None = None
        self._config = load_config()

        tokenizer_model_path = self.model_dir / "spm_bpe_8k.model"
        if not tokenizer_model_path.exists():
            logger.warning(f"Tokenizer model not found at {tokenizer_model_path}. Please run `scripts/prepare_data.py` first.")
            self.tokenizer = None
        else:
            self.tokenizer = SentencePieceTokenizer(tokenizer_model_path)

        last_model_path = self.model_paths["finetune"]
        
        hf_name = os.getenv("HF_MODEL_NAME")
        if hf_name:
            self.model = HFModel(hf_name)
        elif last_model_path.exists() and self.tokenizer:
            try:
                logger.info(f"Loading model from {last_model_path}...")
                self.model = load_transformer(last_model_path)
                if self.model.embed.num_embeddings != self.tokenizer.vocab_size:
                    logger.warning(
                        f"Model vocab size ({self.model.embed.num_embeddings}) and tokenizer vocab size ({self.tokenizer.vocab_size}) mismatch."
                    )
            except Exception as e:
                logger.error(f"Failed to load model from {last_model_path}: {e}")
                self.model = None

    def start_training(self, mode: str) -> Dict[str, Any]:
        '''학습 유형에 따라 분기 처리.'''
        valid, msg = validate_config(self._config)
        if not valid:
            return {"success": False, "msg": msg, "data": None}
        
        if self.tokenizer is None:
            return {"success": False, "msg": "Tokenizer is not initialized. Run `scripts/prepare_data.py`.", "data": None}

        if mode == "pretrain":
            from ..data.subtitle_cleaner import clean_subtitle_files
            clean_subtitle_files(Path("."), self.pretrain_dir)
            dataset = load_pretrain_dataset(self.pretrain_dir)
        elif mode == "additional_finetune":
            dataset = load_instruction_dataset(self.additional_dir)
        else:
            mode = "finetune"
            dataset = load_instruction_dataset(self.finetune_dir)

        logger.info(f"Starting training for mode: {mode}")
        
        if mode == "pretrain":
            trained_model = pretrain(dataset, self._config)
        else:
            trained_model = train_transformer(dataset, self._config, is_pretrain=False)
        
        self.model = trained_model
        
        target_model_path = self.model_paths[mode]
        logger.info(f"Saving trained model to {target_model_path}...")
        save_transformer(self.model, target_model_path)
        
        logger.info("Training complete.")
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
            logger.warning("config save failed: %s", exc)
            return False, str(exc)

    def get_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def delete_model(self) -> bool:
        '''Delete every model file under ``models`` directory.'''
        deleted = False
        for fp in self.model_dir.glob("*.pth"):
            try:
                fp.unlink()
                deleted = True
                logger.info(f"Deleted model file: {fp}")
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.warning("model delete failed: %s", exc)
        if deleted:
            self.model = None
        return deleted

    def infer(self, text: str) -> Dict[str, Any]:
        if not self.model or not self.tokenizer:
            return {"success": False, "msg": "Model or tokenizer not loaded.", "data": None}
        if not text.strip():
            return {"success": False, "msg": "Empty input.", "data": None}
        if len(text) > self.MAX_INPUT_LEN:
            return {"success": False, "msg": "Input text is too long.", "data": None}
        
        if isinstance(self.model, Seq2SeqTransformer):
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            src = src.to(next(self.model.parameters()).device)
            
            out_ids = self.model.generate(
                src,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                pad_id=self.tokenizer.pad_id,
                max_new_tokens=self._config.get("max_response_length", 64),
                temperature=self._config.get("temperature", 0.7),
                top_k=self._config.get("top_k", 0),
                top_p=self._config.get("top_p", 0.9),
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
        '''Apply AutoTuner suggestions to config.'''
        size, tokens, txt_lines, json_lines, skipped = get_dataset_info(
            self.pretrain_dir, self.finetune_dir, self.additional_dir
        )
        logger.info(f"AutoTune triggered: dataset size = {size}, tokens = {tokens}")
        
        cfg = AutoTuner(size, tokens).suggest()
        valid, msg = validate_config(cfg)
        if not valid:
            return {"success": False, "msg": msg, "data": None}
        
        self._config.update(cfg)
        save_config(self._config)
        logger.info("auto-tune applied: %s", cfg)
        return {"success": True, "msg": "auto-tune applied", "data": cfg}
