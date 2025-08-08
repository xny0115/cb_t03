from __future__ import annotations

from pathlib import Path
from typing import List
import sentencepiece as spm

class SentencePieceTokenizer:
    """A wrapper for a custom-trained SentencePiece model."""

    def __init__(self, model_path: Path):
        """
        Initializes the tokenizer by loading a SentencePiece model.

        Args:
            model_path: Path to the trained SentencePiece model file.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found at {model_path}")
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encodes a string into a sequence of token IDs.

        Args:
            text: The input string to encode.
            add_special_tokens: Whether to add BOS and EOS tokens.

        Returns:
            A list of integer token IDs.
        """
        encoded = self.sp.encode_as_ids(text)
        if add_special_tokens:
            return [self.bos_id] + encoded + [self.eos_id]
        return encoded

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a sequence of token IDs back to a string, skipping special tokens.

        Args:
            ids: A list of integer token IDs.

        Returns:
            The decoded string.
        """
        # Filter out special tokens (BOS, EOS, PAD) before decoding
        # This prevents them from being rendered as text.
        special_ids = {self.bos_id, self.eos_id, self.pad_id, self.unk_id}
        return self.sp.decode([id for id in ids if id not in special_ids])

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        """Returns the ID of the padding token."""
        return self.sp.pad_id()

    @property
    def bos_id(self) -> int:
        """Returns the ID of the beginning-of-sentence token."""
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        """Returns the ID of the end-of-sentence token."""
        return self.sp.eos_id()

    @property
    def unk_id(self) -> int:
        """Returns the ID of the unknown token."""
        return self.sp.unk_id()
