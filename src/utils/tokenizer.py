from __future__ import annotations
from pathlib import Path
from typing import List
import sentencepiece as spm

class SentencePieceTokenizer:
    '''A wrapper for a custom-trained SentencePiece model.'''
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found at {model_path}")
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        encoded = self.sp.encode_as_ids(text)
        return [self.bos_id] + encoded + [self.eos_id] if add_special_tokens else encoded

    def decode(self, ids: List[int]) -> str:
        special_ids = {self.bos_id, self.eos_id, self.pad_id, self.unk_id}
        return self.sp.decode([id for id in ids if id not in special_ids])

    @property
    def vocab_size(self) -> int: return self.sp.get_piece_size()
    @property
    def pad_id(self) -> int: return self.sp.pad_id()
    @property
    def bos_id(self) -> int: return self.sp.bos_id()
    @property
    def eos_id(self) -> int: return self.sp.eos_id()
    @property
    def unk_id(self) -> int: return self.sp.unk_id()
