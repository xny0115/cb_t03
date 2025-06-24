from __future__ import annotations

"""Pre-tokenize dataset into torch shards."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

from src.data.loader import QADataset
from src.utils.vocab import build_vocab, encode_tokens


def _hash(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for p in paths:
        with open(p, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()[:8]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize dataset")
    parser.add_argument("src", type=Path)
    parser.add_argument("--out", type=Path, default=Path("cache"))
    args = parser.parse_args()

    ds = QADataset(args.src)
    vocab = build_vocab(ds)
    tokhash = hashlib.sha1(b"komoran").hexdigest()[:8]
    rawhash = _hash(ds.paths)
    out_dir = args.out / f"{tokhash}-{rawhash}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vocab, open(out_dir / "vocab.json", "w"), ensure_ascii=False)

    shard_size = 10000
    shard: list[tuple[torch.Tensor, torch.Tensor]] = []
    idx = 0
    for pair in ds.pairs:
        q = encode_tokens(pair.tokens_q, pair.concepts, pair.domain, vocab)
        a = encode_tokens(pair.tokens_a, pair.concepts, pair.domain, vocab)
        shard.append((q, a))
        if len(shard) >= shard_size:
            torch.save(shard, out_dir / f"shard_{idx:05}.pt", _use_new_zipfile_serialization=False)
            shard = []
            idx += 1
    if shard:
        torch.save(shard, out_dir / f"shard_{idx:05}.pt", _use_new_zipfile_serialization=False)


if __name__ == "__main__":  # pragma: no cover
    main()
