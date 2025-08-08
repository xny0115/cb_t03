from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging, os, torch
from ..data.loader import load_instruction_dataset, load_pretrain_dataset
from ..config import load_config, save_config
from ..model import Seq2SeqTransformer, save_transformer, load_transformer
from ..training.simple import train as train_transformer, pretrain
from ..utils.tokenizer import SentencePieceTokenizer
from ..utils.validator import validate_config

logger = logging.getLogger(__name__)

class ChatbotService:
    '''Instruction 기반 챗봇 서비스.'''
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
        """
        UI 초기화용 설정 딕셔너리 반환
        필수 키: "model_path", "device"
        """
        return self._config

    def start_training(self, mode: str) -> Dict[str, Any]:
        if self.tokenizer is None:
            return {"success": False, "msg": "Tokenizer not initialized."}
        
        if mode == "pretrain":
            dataset = load_pretrain_dataset(self.data_dir / "pretrain")
            trained_model = pretrain(dataset, self._config, model=self.model)
        else:
            dataset = load_instruction_dataset(self.data_dir / ("additional_finetune" if mode == "additional_finetune" else "finetune"))
            trained_model = train_transformer(dataset, self._config, is_pretrain=False, model=self.model)
        
        self.model = trained_model
        target_model_path = self.model_dir / f"{mode}.pth"
        logger.info(f"Saving trained model to {target_model_path}...")
        save_transformer(self.model, target_model_path)
        return {"success": True, "msg": "Training complete."}

    def infer(self, text: str) -> Dict[str, Any]:
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
