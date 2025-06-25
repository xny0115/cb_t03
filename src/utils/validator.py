"""Configuration validation utilities."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

# 필수 하이퍼파라미터 키 목록
REQUIRED_KEYS = [
    "batch_size",
    "model_dim",
    "ff_dim",
    "num_encoder_layers",
    "num_decoder_layers",
    "num_epochs",
    "learning_rate",
    "use_mixed_precision",
]


def validate_config(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """Return True if all required keys exist and values are valid."""
    for key in REQUIRED_KEYS:
        if key not in cfg:
            return False, f"{key} missing"
    for k, v in cfg.items():
        if v is None:
            return False, f"{k} is None"
        if isinstance(v, str) and v.strip() == "":
            return False, f"{k} is empty"
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return False, f"{k} invalid"
    return True, ""
