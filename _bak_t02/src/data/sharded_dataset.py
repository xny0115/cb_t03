from __future__ import annotations

"""Iterable dataset loading pretokenized shards."""
import random
from pathlib import Path
from torch.utils.data import IterableDataset, get_worker_info
import torch


class ShardedDataset(IterableDataset):
    def __init__(self, shard_root: Path) -> None:
        self.shards = sorted(shard_root.glob("shard_*.pt"))
        random.shuffle(self.shards)

    def __iter__(self):
        info = get_worker_info()
        world = info.num_workers if info else 1
        wid = info.id if info else 0
        for shard in self.shards[wid::world]:
            data = torch.load(shard, map_location="cpu")
            for item in data:
                yield item
