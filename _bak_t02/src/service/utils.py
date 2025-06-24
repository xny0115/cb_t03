from __future__ import annotations

"""Helper utilities for ChatbotService."""
from typing import Any, Dict
from ..config import Config

_CFG_MAP = {
    "num_epochs": ("epochs", 20),
    "batch_size": ("batch_size", 32),
    "learning_rate": ("lr", 1e-3),
    "dropout_ratio": ("dropout_ratio", 0.1),
    "warmup_steps": ("warmup_steps", 0),
    "max_sequence_length": ("max_sequence_length", 128),
    "num_heads": ("num_heads", 4),
    "num_encoder_layers": ("num_encoder_layers", 2),
    "num_decoder_layers": ("num_decoder_layers", 2),
    "model_dim": ("model_dim", 128),
    "ff_dim": ("ff_dim", 512),
    "top_k": ("top_k", 10),
    "top_p": ("top_p", 0.9),
    "no_repeat_ngram": ("no_repeat_ngram", 2),
    "temperature": ("temperature", 0.7),
    "model_type": ("model_type", "transformer_base"),
    "gradient_clipping": ("gradient_clipping", 1.0),
    "weight_decay": ("weight_decay", 0.01),
    "early_stopping": ("early_stopping", True),
    "early_stopping_patience": ("early_stopping_patience", 8),
    "save_every": ("save_every", 0),
    "num_workers": ("num_workers", 4),
    "pin_memory": ("pin_memory", True),
    "use_mixed_precision": ("use_mixed_precision", False),
    "repetition_penalty": ("repetition_penalty", 1.1),
    "max_response_length": ("max_response_length", 64),
    "data_preprocessing": ("data_preprocessing", "none"),
    "embedding_dim": ("embedding_dim", 256),
    "activation_function": ("activation_function", "relu"),
    "optimizer_selection": ("optimizer_selection", "adam"),
    "lr_scheduler": ("lr_scheduler", "none"),
    "normalization_technique": ("normalization_technique", "layer_norm"),
    "attention_type": ("attention_type", "multi_head"),
    "positional_encoding": ("positional_encoding", "sine"),
    "pattern_recognition": ("pattern_recognition", False),
    "beam_search_size": ("beam_search_size", 1),
    "diversity_penalty": ("diversity_penalty", 0.0),
    "verbose": ("verbose", False),
}


def to_config(data: Dict[str, Any]) -> Config:
    """Convert config dict to Config dataclass."""
    return Config(**{k: data.get(src, d) for k, (src, d) in _CFG_MAP.items()})


def simple_fallback(prompt: str) -> str:
    """Return short apology."""
    return (
        "죄송합니다. 답변을 준비하지 못했습니다."
        if "?" in prompt
        else "답변을 생성하지 못했습니다."
    )


def normalize_config(cfg: Config) -> Config:
    """Return ``cfg`` with safe values."""
    cfg.num_epochs = max(1, int(cfg.num_epochs))
    cfg.batch_size = max(1, int(cfg.batch_size))
    cfg.learning_rate = max(1e-6, float(cfg.learning_rate))
    cfg.dropout_ratio = min(max(float(cfg.dropout_ratio), 0.0), 0.9)
    cfg.warmup_steps = max(0, int(cfg.warmup_steps))
    cfg.max_sequence_length = max(16, int(cfg.max_sequence_length))
    cfg.num_heads = max(1, int(cfg.num_heads))
    cfg.num_encoder_layers = max(1, int(cfg.num_encoder_layers))
    cfg.num_decoder_layers = max(1, int(cfg.num_decoder_layers))
    cfg.model_dim = max(16, int(cfg.model_dim))
    cfg.ff_dim = max(16, int(cfg.ff_dim))
    cfg.top_k = max(1, int(cfg.top_k))
    cfg.top_p = min(max(float(cfg.top_p), 0.0), 0.99)
    cfg.no_repeat_ngram = max(1, int(cfg.no_repeat_ngram))
    cfg.temperature = min(max(float(cfg.temperature), 0.1), 5.0)
    cfg.model_type = str(cfg.model_type)
    cfg.gradient_clipping = max(0.0, float(cfg.gradient_clipping))
    cfg.weight_decay = min(max(float(cfg.weight_decay), 0.0), 1.0)
    cfg.early_stopping = bool(cfg.early_stopping)
    cfg.early_stopping_patience = max(1, int(cfg.early_stopping_patience))
    cfg.save_every = max(0, int(cfg.save_every))
    cfg.num_workers = max(0, int(cfg.num_workers))
    cfg.pin_memory = bool(cfg.pin_memory)
    cfg.use_mixed_precision = bool(cfg.use_mixed_precision)
    cfg.repetition_penalty = max(1.0, float(cfg.repetition_penalty))
    cfg.max_response_length = max(1, int(cfg.max_response_length))
    cfg.data_preprocessing = str(cfg.data_preprocessing)
    cfg.embedding_dim = max(8, int(cfg.embedding_dim))
    cfg.activation_function = str(cfg.activation_function)
    cfg.optimizer_selection = str(cfg.optimizer_selection)
    cfg.lr_scheduler = str(cfg.lr_scheduler)
    cfg.normalization_technique = str(cfg.normalization_technique)
    cfg.attention_type = str(cfg.attention_type)
    cfg.positional_encoding = str(cfg.positional_encoding)
    cfg.pattern_recognition = bool(cfg.pattern_recognition)
    cfg.beam_search_size = max(1, int(cfg.beam_search_size))
    cfg.diversity_penalty = max(0.0, float(cfg.diversity_penalty))
    return cfg
