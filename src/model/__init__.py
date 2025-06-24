from .dummy import DummyModel, load_model, save_model
try:  # pragma: no cover - optional dependency
    from .hf_model import HFModel
except Exception:  # pragma: no cover - best effort
    class HFModel:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("HFModel unavailable")
from .transformer import (
    Seq2SeqTransformer,
    save_transformer,
    load_transformer,
)

__all__ = [
    "DummyModel",
    "load_model",
    "save_model",
    "HFModel",
    "Seq2SeqTransformer",
    "save_transformer",
    "load_transformer",
]
