from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from ..data.loader import InstructionSample
from ..model.transformer import Seq2SeqTransformer
from ..utils.tokenizer import CharTokenizer


class _PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[List[int], List[int]]]):
        self.data = pairs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def _collate(batch):
    srcs, tgts = zip(*batch)
    srcs = [torch.tensor(s, dtype=torch.long) for s in srcs]
    tgts = [torch.tensor(t, dtype=torch.long) for t in tgts]
    src_pad = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    return src_pad, tgt_pad


def train(samples: List[InstructionSample], epochs: int = 5, lr: float = 1e-3):
    texts = [f"{s.instruction} {s.input} {s.output}" for s in samples]
    tokenizer = CharTokenizer(texts)
    pairs = []
    for s in samples:
        src = tokenizer.encode(f"{s.instruction} {s.input}".strip(), True)
        tgt = tokenizer.encode(s.output, True)
        pairs.append((src, tgt))

    dataset = _PairDataset(pairs)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=_collate)

    model = Seq2SeqTransformer(tokenizer.vocab_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()
            out = model(src, tgt[:, :-1])
            loss = crit(out.reshape(-1, tokenizer.vocab_size), tgt[:, 1:].reshape(-1))
            loss.backward()
            opt.step()

    return model, tokenizer
