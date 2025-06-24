from __future__ import annotations
from pathlib import Path
import logging
import json
import torch
from ..data.loader import QADataset
from ..utils.vocab import build_vocab, Tokenizer
from ..tuning.auto import AutoTuner
from ..model.transformer import Seq2SeqTransformer

logger = logging.getLogger(__name__)


def auto_tune(cfg: dict) -> None:  # pragma: no cover
    try:
        ds = QADataset(Path("datas"))
        sugg = AutoTuner(len(ds)).suggest_config()
        for k, v in sugg.__dict__.items():
            if k in cfg:
                cfg[k] = v
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Auto tune failed: %s", exc)


def load_model(path: Path) -> tuple[Tokenizer, Seq2SeqTransformer]:  # pragma: no cover
    vocab_path = path.with_suffix(".vocab.json")
    if vocab_path.exists():
        vocab = json.load(open(vocab_path))
    else:
        ds = QADataset(Path("datas"))
        vocab = build_vocab(ds)
    tokenizer = Tokenizer(vocab)
    model = Seq2SeqTransformer(vocab_size=len(vocab))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return tokenizer, model
