"""Simple Transformer Seq2Seq model."""

from __future__ import annotations

import math
import logging
from typing import Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


class Seq2SeqTransformer(nn.Module):
    """Minimal Seq2Seq Transformer implementation."""

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
        self.model_type = "Transformer"
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
        """Run full transformer forward pass."""
        src = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        tgt = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)
        output = self.transformer(src, tgt)
        return self.fc_out(output)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_new_tokens: int = 64,
        max_length: int | None = None,
        eos_id: int = 1,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        no_repeat_ngram: int = 2,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Generate sequence with sampling and n-gram blocking."""
        self.eval()
        if max_length is not None:
            max_new_tokens = max_length
        device = src.device
        ys = torch.tensor([[eos_id]], device=device)
        ngrams: set[tuple[int, ...]] = set()
        for step in range(max_new_tokens):
            out = self(src, ys)[:, -1, :] / temperature
            out = out.squeeze(0)
            if repetition_penalty != 1.0:
                uniq = torch.unique(ys)
                out[uniq] = out[uniq] / repetition_penalty
            if no_repeat_ngram > 1 and ys.size(1) >= no_repeat_ngram:
                prefix = ys[0, -no_repeat_ngram + 1 :].tolist()
                banned = [w for w in range(out.size(0)) if tuple(prefix + [w]) in ngrams]
                if banned:
                    out[banned] = -float("inf")
            k = min(top_k, out.size(0))
            topk_val, topk_idx = out.topk(k)
            prob = torch.softmax(topk_val, dim=-1)
            if torch.sum(prob) <= 0 or not torch.isfinite(prob).all():
                logger.warning("multinomial fallback used at step %d", step)
                next_id = topk_idx[0].view(1, 1)
            elif 0.0 < top_p < 1.0:
                s_probs, s_idx = prob.sort(descending=True)
                cum = s_probs.cumsum(dim=-1)
                mask = cum > top_p
                if mask.all():
                    mask[-1] = False
                s_probs[mask] = 0
                s_probs.div_(s_probs.sum())
                choice = torch.multinomial(s_probs, 1).item()
                idx = s_idx[choice].item()
                next_id = topk_idx[idx].view(1, 1)
            else:
                choice = torch.multinomial(prob, 1).item()
                next_id = topk_idx[choice].view(1, 1)
            ys = torch.cat([ys, next_id], dim=1)
            if no_repeat_ngram > 1 and ys.size(1) >= no_repeat_ngram:
                ngrams.add(tuple(ys[0, -no_repeat_ngram:].tolist()))
            if next_id.item() == eos_id:
                break
        return ys


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(
        self, emb_size: int, dropout: float = 0.1, max_len: int = 5000
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size)
        )
        pe = torch.zeros(max_len, 1, emb_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to tensor."""
        if x.dim() != 3:
            raise ValueError("positional encoding expects 3D input")
        return self.dropout(x + self.pe[: x.size(1)].transpose(0, 1))
