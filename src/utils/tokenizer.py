from __future__ import annotations

from typing import List
from pathlib import Path
import json


class CharTokenizer:
    """단순 문자 단위 토크나이저."""

    def __init__(self, texts: List[str]) -> None:
        chars = sorted({ch for t in texts for ch in t})
        self.stoi = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
        for i, ch in enumerate(chars, start=3):
            self.stoi[ch] = i
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_vocab(cls, vocab: dict[str, int]) -> "CharTokenizer":
        obj = cls([])
        obj.stoi = vocab
        obj.itos = {i: ch for ch, i in vocab.items()}
        return obj

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        ids = [self.stoi.get(ch, 3) for ch in text]
        if add_special:
            return [self.stoi["<bos>"]] + ids + [self.stoi["<eos>"]]
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids if i >= 3)

    def save(self, path: str | Path) -> None:
        """토크나이저 vocabulary를 JSON 파일로 저장한다."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
