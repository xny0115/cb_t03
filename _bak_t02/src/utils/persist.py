from __future__ import annotations

"""Simple JSON persistence helpers."""

from pathlib import Path
from typing import Any, Dict
import json

DEFAULT_PATH = Path("configs/current.json")


def load_json(path: Path = DEFAULT_PATH) -> Dict[str, Any]:
    """Load JSON data from file."""
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    return {}


def save_json(data: Dict[str, Any], path: Path = DEFAULT_PATH) -> None:
    """Write JSON data to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
