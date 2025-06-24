"""Data pipeline package."""

from pathlib import Path
import os
import json

from .loader import QADataset
from .sharded_dataset import ShardedDataset

_CFG_FILE = Path("configs/current.json")
if _CFG_FILE.exists():
    cfg = json.load(open(_CFG_FILE))
else:
    cfg = {}
cfg["dataset_mode"] = os.getenv("DATASET_MODE", cfg.get("dataset_mode", "small"))
cfg["shard_root"] = os.getenv("SHARD_ROOT", cfg.get("shard_root", "cache"))


class SmallInMemoryDataset(QADataset):
    """Alias for backward compatibility."""


if cfg["dataset_mode"] == "full":
    root = Path(cfg["shard_root"])
    dirs = list(root.glob("*-*"))
    latest = max(dirs) if dirs else root
    dataset = ShardedDataset(latest)
else:
    dataset = SmallInMemoryDataset(Path("datas"))

__all__ = ["QADataset", "ShardedDataset", "dataset", "SmallInMemoryDataset"]
