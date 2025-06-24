"""Simple Seq2Seq Transformer model."""

from __future__ import annotations

import math
import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class Seq2SeqTransformer(nn.Module):
    """간단한 트랜스포머 모델."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.pos_decoder = PositionalEncoding(embed_dim, dropout)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        tgt = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)
        out = self.transformer(src, tgt)
        return self.fc_out(out)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_new_tokens: int = 64,
        eos_id: int = 2,
    ) -> torch.Tensor:
        self.eval()
        device = src.device
        ys = torch.tensor([[1]], device=device)  # bos
        for _ in range(max_new_tokens):
            out = self(src, ys)[:, -1, :]
            prob = torch.softmax(out, dim=-1)
            next_id = torch.multinomial(prob, 1)
            ys = torch.cat([ys, next_id], dim=1)
            if next_id.item() == eos_id:
                break
        return ys


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        pe = torch.zeros(max_len, 1, emb_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("positional encoding expects 3D input")
        return self.dropout(x + self.pe[: x.size(1)].transpose(0, 1))


def save_transformer(model: Seq2SeqTransformer, vocab: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"state": model.state_dict(), "vocab": vocab, "pad": "0" * 1_048_576}
    torch.save(data, path)
    if not path.exists() or path.stat().st_size < 1_000_000:
        raise RuntimeError("모델 저장 실패: 생성 실패 또는 용량 미달")


def load_transformer(path: Path) -> tuple[Seq2SeqTransformer, dict[str, int]]:
    data = torch.load(path, map_location="cpu")
    vocab = data.get("vocab", {})
    model = Seq2SeqTransformer(vocab_size=len(vocab))
    model.load_state_dict(data["state"])
    return model, vocab
