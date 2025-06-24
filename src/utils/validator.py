"""Configuration validation utilities."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple


def validate_config(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """Return True if all values are defined and not NaN."""
    for k, v in cfg.items():
        if v is None:
            return False, f"{k} is None"
        if isinstance(v, str) and v.strip() == "":
            return False, f"{k} is empty"
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return False, f"{k} invalid"
    return True, ""
