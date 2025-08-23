"""Project configuration helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "configs/current.json"))

DEFAULT_CONFIG: Dict[str, Any] = {
    "num_epochs": 20,
    "batch_size": 48,
    "learning_rate": 2e-4,
    "dropout_ratio": 0.1,
    "grad_clip": 1.0,
    "min_lr": 1e-5,
    "warmup_steps": 0,
    "max_sequence_length": 128,
    "num_heads": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "model_dim": 256,
    "ff_dim": 1024,
    "top_k": 10,
    "temperature": 0.7,
    "top_p": 0.9,
    "data_preprocessing": "none",
    "embedding_dim": 256,
    "activation_function": "relu",
    "optimizer_selection": "adam",
    "lr_scheduler": "none",
    "normalization_technique": "layer_norm",
    "attention_type": "multi_head",
    "positional_encoding": "sine",
    "gradient_clipping": 1.0,
    "weight_decay": 0.01,
    "early_stopping": True,
    "early_stopping_patience": 8,
    "save_every": 0,
    "num_workers": 6,
    "pin_memory": True,
    "use_mixed_precision": True,
    "repetition_penalty": 1.1,
    "max_response_length": 64,
    "beam_search_size": 3,
    "diversity_penalty": 0.5,
    "pattern_recognition": False,
    "spm_model_path": "tokenizer/spm.model",
    "resume": False,
}


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            data = json.load(open(CONFIG_PATH, encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({k: data.get(k, v) for k, v in DEFAULT_CONFIG.items()})
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(CONFIG_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
