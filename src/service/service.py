from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging, os, torch
from ..data.loader import load_instruction_dataset, load_pretrain_dataset
from ..config import load_config, save_config
from ..model import Seq2SeqTransformer, save_transformer, load_transformer
from ..training.simple import train as train_transformer, pretrain
from ..utils.tokenizer import SentencePieceTokenizer
from ..utils.validator import validate_config, REQUIRED_KEYS
from ..utils.logger import setup_logger
setup_logger()  # 서비스 레벨에서 로그 초기화

logger = logging.getLogger(__name__)

class ChatbotService:
    '''Instruction 기반 챗봇 서비스.'''
    MAX_INPUT_LEN = 512

    def __init__(self) -> None:
        self.model_dir = Path("models")
        self.data_dir = Path("datas")
        self._config = load_config()
        self.tokenizer = None
        self.model: Optional[Seq2SeqTransformer] = None
        
        tokenizer_path = Path(self._config.get("tokenizer_path", "models/spm_bpe_8k.model"))
        if tokenizer_path.exists():
            self.tokenizer = SentencePieceTokenizer(tokenizer_path)
        else:
            logger.warning(f"Tokenizer not found at {tokenizer_path}. Run `scripts/prepare_data.py` first.")

        model_files = [p for p in self.model_dir.glob("*.pth") if 'checkpoint' not in p.name]
        if model_files:
            latest_model_path = max(model_files, key=os.path.getmtime)
            if self.tokenizer:
                try:
                    logger.info(f"Loading most recent model from {latest_model_path}...")
                    self.model = load_transformer(latest_model_path)
                    if self.model.embed.num_embeddings != self.tokenizer.vocab_size:
                        logger.warning(f"Model and tokenizer vocab size mismatch. Re-training might be necessary.")
                except Exception as e:
                    logger.error(f"Failed to load model from {latest_model_path}: {e}")

    def get_config(self) -> dict:
        """UI 초기화용 설정 딕셔너리 반환"""
        return self._config

    def auto_tune(self) -> Dict[str, Any]:
        ok, msg = validate_config(self._config)
        if not ok:
            return {"success": False, "msg": msg}
        tuned = {k: self._config[k] for k in REQUIRED_KEYS}
        save_config(self._config)
        return {"success": True, "data": tuned}

    def start_training(self, mode: str) -> Dict[str, Any]:
        """Run training in the specified mode.

        Args:
            mode: 학습 모드('pretrain', 'finetune', 'additional_finetune').

        Returns:
            학습 성공 여부 및 메시지.
        """
        ok, msg = validate_config(self._config)
        if not ok:
            return {"success": False, "msg": msg}

        resume = bool(self._config.get("resume", False))
        from pathlib import Path
        ckpt = Path(self._config.get("checkpoint_path", "models/training_state.pth"))
        resume = ckpt.exists()

        import platform, torch, logging
        logger = logging.getLogger(__name__)

        # 환경 보정
        if platform.system() == "Windows":
            self._config["num_workers"] = 0
            self._config["pin_memory"] = False
        if not torch.cuda.is_available():
            self._config["use_mixed_precision"] = False

        # 토크나이저 확인
        if self.tokenizer is None:
            return {"success": False, "msg": "Tokenizer not initialized."}

        # 데이터 로딩
        if mode == "pretrain":
            dataset = load_pretrain_dataset(self.data_dir / "pretrain")
        else:
            dataset = load_instruction_dataset(
                self.data_dir
                / ("additional_finetune" if mode == "additional_finetune" else "finetune")
            )
        ds_len = len(dataset) if hasattr(dataset, "__len__") else 0
        logger.info("start_training mode=%s dataset_size=%d", mode, ds_len)
        if ds_len < 2:
            return {"success": False, "msg": f"dataset too small: {ds_len} samples"}

        if mode == "pretrain":
            trained_model = pretrain(dataset, self._config, model=self.model, resume=resume)
        else:
            trained_model = train_transformer(
                dataset,
                self._config,
                is_pretrain=False,
                model=self.model,
                resume=resume,
            )

        self.model = trained_model
        self.dataset = dataset
        target_model_path = self.model_dir / f"{mode}.pth"
        logger.info(f"Saving trained model to {target_model_path}...")
        save_transformer(self.model, target_model_path)
        return {"success": True, "msg": "Training complete."}

    def infer(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"success": False, "msg": "empty_input"}
        if len(text) > self.MAX_INPUT_LEN:
            return {"success": False, "msg": "too_long"}
        if self.model and hasattr(self.model, "predict"):
            out = self.model.predict("", text)
            return {"success": bool(out), "msg": out or "no_answer"}
        if not (self.model and self.tokenizer):
            return {"success": False, "msg": "Model or tokenizer not loaded."}
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(next(self.model.parameters()).device)
        out_ids = self.model.generate(
            src,
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            pad_id=self.tokenizer.pad_id,
            max_new_tokens=self._config.get("max_response_length", 64),
            temperature=float(self._config.get("temperature", 0.7)),
            top_k=int(self._config.get("top_k", 0)),
            top_p=float(self._config.get("top_p", 0.9)),
        )
        return {"success": True, "data": self.tokenizer.decode(out_ids.squeeze().tolist())}

    def delete_model(self) -> bool:
        """Delete trained model files except checkpoint and tokenizer."""
        removed = False
        for p in self.model_dir.glob("*.pth"):
            if p.name == "training_state.pth":
                continue
            p.unlink(missing_ok=True)
            removed = True
        self.model = None
        return removed
