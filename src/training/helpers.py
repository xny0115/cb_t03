from __future__ import annotations

from typing import List, Tuple

import logging
import time
import torch
from torch import nn
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[List[int], List[int]]]):
        self.data = pairs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def collate(batch):
    srcs, tgts = zip(*batch)
    srcs = [torch.as_tensor(s, dtype=torch.long) for s in srcs]
    tgts = [torch.as_tensor(t, dtype=torch.long) for t in tgts]
    src_pad = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    return src_pad, tgt_pad


def timed_collate(batch):
    start = time.perf_counter()
    out = collate(batch)
    ms = (time.perf_counter() - start) * 1000
    logging.getLogger(__name__).debug("collate_fn time: %.2fms", ms)
    return out


def log_dataset_stats(texts: List[str]) -> None:
    logger = logging.getLogger(__name__)
    non_empty = [t for t in texts if t.strip()]
    dup_lines = len(non_empty) - len(set(non_empty))
    if non_empty:
        avg_chars = sum(len(t) for t in non_empty) / len(non_empty)
        max_chars = max(len(t) for t in non_empty)
        min_chars = min(len(t) for t in non_empty)
    else:
        avg_chars = max_chars = min_chars = 0
    logger.debug(
        "dataset lines=%d empty=%d dup=%d(%.2f) avg=%.2f max=%d min=%d",
        len(texts),
        len(texts) - len(non_empty),
        dup_lines,
        dup_lines / len(non_empty) if non_empty else 0,
        avg_chars,
        max_chars,
        min_chars,
    )


__all__ = ["PairDataset", "collate", "timed_collate", "log_dataset_stats"]
