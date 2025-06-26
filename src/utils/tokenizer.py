from __future__ import annotations

from typing import List
from pathlib import Path
import json


class CharTokenizer:
    """단순 문자 단위 토크나이저."""

    SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, texts: List[str]) -> None:
        chars = sorted({ch for t in texts for ch in t})
        self.stoi = {tok: i for i, tok in enumerate(self.SPECIALS)}
        for i, ch in enumerate(chars, start=len(self.SPECIALS)):
            self.stoi[ch] = i
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_vocab(cls, vocab: dict[str, int]) -> "CharTokenizer":
        obj = cls([])
        obj.stoi = vocab
        obj.itos = {i: ch for ch, i in vocab.items()}
        return obj

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """텍스트를 id 시퀀스로 변환한다. 미등록 문자는 ``<unk>``으로 매핑한다."""
        unk = self.stoi["<unk>"]
        ids = [self.stoi.get(ch, unk) for ch in text]
        if add_special:
            return [self.stoi["<bos>"]] + ids + [self.stoi["<eos>"]]
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids if i >= len(self.SPECIALS))

    def save(self, path: str | Path) -> None:
        """토크나이저 vocabulary를 JSON 파일로 저장한다."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        """패딩 토큰의 ID를 반환한다."""
        return self.stoi["<pad>"]
