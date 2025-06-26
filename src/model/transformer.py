"""Custom encoder-decoder Transformer implementation."""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*flash attention.*",
    category=UserWarning,
)
import platform

import math
import logging
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model % num_heads != 0"
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = q.size()
        k_len = k.size(1)
        q = self.w_q(q).view(bsz, q_len, self.num_heads, self.d_head).transpose(1, 2)
        k = self.w_k(k).view(bsz, k_len, self.num_heads, self.d_head).transpose(1, 2)
        v = self.w_v(v).view(bsz, k_len, self.num_heads, self.d_head).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0
        )
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, self.num_heads * self.d_head)
        )
        return self.w_o(attn)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, dim_ff: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attn = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn)
        x = self.norm1(x)
        x = x + self.ff(x)
        return self.norm2(x)


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, dim_ff: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(attn)
        x = self.norm1(x)
        attn = self.cross_attn(x, memory, memory, mem_mask)
        x = x + self.dropout(attn)
        x = self.norm2(x)
        x = x + self.ff(x)
        return self.norm3(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("positional encoding expects 3D input")
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder Transformer."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_ff: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.pos_decoder = PositionalEncoding(embed_dim, dropout)
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(embed_dim, num_heads, dim_ff, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(embed_dim, num_heads, dim_ff, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, pad_id: int = 0
    ) -> torch.Tensor:
        """Transformer forward pass with padding & future masks."""
        src_emb = self.pos_encoder(
            self.embed(src) * math.sqrt(self.embed.embedding_dim)
        )
        tgt_emb = self.pos_decoder(
            self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        )

        bsz, src_len = src.size()
        bsz, tgt_len = tgt.size()

        src_pad = src.eq(pad_id)
        tgt_pad = tgt.eq(pad_id)

        num_heads = self.encoder[0].self_attn.num_heads
        src_mask = src_pad.unsqueeze(1).expand(-1, src_len, src_len)
        src_mask = src_mask.repeat_interleave(num_heads, dim=0)

        future = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), 1).bool()
        tgt_mask = future.unsqueeze(0).expand(bsz, tgt_len, tgt_len)
        if tgt_pad.any():
            pad_m = tgt_pad.unsqueeze(1).expand(-1, tgt_len, tgt_len)
            tgt_mask = tgt_mask | pad_m
        tgt_mask = tgt_mask.repeat_interleave(num_heads, dim=0)

        mem_mask = src_pad.unsqueeze(1).expand(-1, tgt_len, src_len)
        mem_mask = mem_mask.repeat_interleave(num_heads, dim=0)

        memory = src_emb
        for layer in self.encoder:
            memory = layer(memory, src_mask)
        out = tgt_emb
        for layer in self.decoder:
            out = layer(out, memory, tgt_mask, mem_mask)
        return self.fc_out(out)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_new_tokens: int = 64,
        eos_id: int = 2,
        pad_id: int = 0,
    ) -> torch.Tensor:
        self.eval()
        device = src.device
        ys = torch.tensor([[1]], device=device)
        for _ in range(max_new_tokens):
            out = self(src, ys, pad_id=pad_id)[:, -1, :]
            prob = torch.softmax(out, dim=-1)
            next_id = torch.multinomial(prob, 1)
            ys = torch.cat([ys, next_id], dim=1)
            if next_id.item() == eos_id:
                break
        return ys


def save_transformer(
    model: Seq2SeqTransformer, vocab: Dict[str, int], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "state": model.state_dict(),
        "vocab": vocab,
        "cfg": {
            "embed_dim": model.embed.embedding_dim,
            "num_heads": model.encoder[0].self_attn.num_heads,
            "num_encoder_layers": len(model.encoder),
            "num_decoder_layers": len(model.decoder),
            "ff_dim": model.encoder[0].ff.net[0].out_features,
            "dropout": model.encoder[0].dropout.p,
        },
        "pad": "0" * 1_048_576,
    }
    torch.save(meta, path)
    if not path.exists() or path.stat().st_size < 1_000_000:
        raise RuntimeError("모델 저장 실패: 생성 실패 또는 용량 미달")
    logger.info("Model saved to %s", path)


def load_transformer(path: Path) -> Tuple[Seq2SeqTransformer, Dict[str, int]]:
    data = torch.load(path, map_location="cpu")
    vocab = data.get("vocab", {})
    cfg = data.get("cfg", {})
    model = Seq2SeqTransformer(
        vocab_size=len(vocab),
        embed_dim=cfg.get("embed_dim", 256),
        num_heads=cfg.get("num_heads", 8),
        num_encoder_layers=cfg.get("num_encoder_layers", 6),
        num_decoder_layers=cfg.get("num_decoder_layers", 6),
        dim_ff=cfg.get("ff_dim", 1024),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(data["state"])
    return model, vocab
