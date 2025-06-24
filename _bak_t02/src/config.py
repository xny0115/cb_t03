from __future__ import annotations

"""Application configuration management."""

from pathlib import Path
from typing import Any, Dict
import json
import os
from dataclasses import dataclass

from .utils.persist import load_json
from .config_schema import CONFIG_SCHEMA

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "configs/current.json"))


def apply_schema(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing keys using ``CONFIG_SCHEMA`` and cast types."""
    for key, (typ, default) in CONFIG_SCHEMA.items():
        if key in cfg:
            try:
                cfg[key] = typ(cfg[key])
            except Exception:
                cfg[key] = default
        else:
            cfg[key] = default
    return cfg

DEFAULT_CONFIG: Dict[str, Any] = {
    "epochs": 20,
    "batch_size": 32,
    "lr": 1e-3,
    "dropout_ratio": 0.1,
    "warmup_steps": 0,
    "max_sequence_length": 128,
    "num_heads": 4,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "model_dim": 128,
    "ff_dim": 512,
    "top_k": 10,
    "top_p": 0.9,
    "no_repeat_ngram": 2,
    "temperature": 0.7,
    "model_type": "transformer_base",
    "gradient_clipping": 1.0,
    "weight_decay": 0.01,
    "early_stopping": True,
    "early_stopping_patience": 8,
    "save_every": 0,
    "num_workers": 4,
    "pin_memory": True,
    "use_mixed_precision": False,
    "repetition_penalty": 1.1,
    "max_response_length": 64,
    "data_preprocessing": "none",
    "embedding_dim": 256,
    "activation_function": "relu",
    "optimizer_selection": "adam",
    "lr_scheduler": "none",
    "normalization_technique": "layer_norm",
    "attention_type": "multi_head",
    "positional_encoding": "sine",
    "pattern_recognition": False,
    "beam_search_size": 1,
    "diversity_penalty": 0.0,
    "verbose": False,
    "force_gpu": False,
}


@dataclass
class Config:
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    dropout_ratio: float = 0.1
    warmup_steps: int = 0
    max_sequence_length: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    model_dim: int = 128
    ff_dim: int = 512
    top_k: int = 10
    top_p: float = 0.9
    no_repeat_ngram: int = 2
    temperature: float = 0.7
    model_type: str = "transformer_base"
    gradient_clipping: float = 1.0
    weight_decay: float = 0.01
    early_stopping: bool = True
    early_stopping_patience: int = 8
    save_every: int = 0
    num_workers: int = 4
    pin_memory: bool = True
    use_mixed_precision: bool = False
    repetition_penalty: float = 1.1
    max_response_length: int = 64
    data_preprocessing: str = "none"
    embedding_dim: int = 256
    activation_function: str = "relu"
    optimizer_selection: str = "adam"
    lr_scheduler: str = "none"
    normalization_technique: str = "layer_norm"
    attention_type: str = "multi_head"
    positional_encoding: str = "sine"
    pattern_recognition: bool = False
    beam_search_size: int = 1
    diversity_penalty: float = 0.0
    verbose: bool = False


def load_config() -> Dict[str, Any]:
    """Load configuration from disk."""
    data = load_json(CONFIG_PATH)
    cfg = DEFAULT_CONFIG.copy()
    if data:
        cfg.update(data)
    cfg = apply_schema(cfg)
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    """Persist configuration to disk."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(apply_schema(cfg), open(CONFIG_PATH, "w"), indent=2)
