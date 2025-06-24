import pytest
pytest.importorskip("torch")
import torch
from src.model.transformer import Seq2SeqTransformer

pytestmark = pytest.mark.requires_torch

class DummyModel(Seq2SeqTransformer):
    def __init__(self):
        super().__init__(vocab_size=5)
    def forward(self, src, tgt):
        out = torch.full((tgt.size(0), src.size(1), 5), float('-inf'))
        return out

def test_multinomial_fallback(caplog):
    model = DummyModel()
    src = torch.tensor([[0]])
    out = model.generate(src, max_new_tokens=3)
    ids = out.view(-1).tolist()[1:]
    assert ids != []
    assert any('multinomial fallback used' in r.message for r in caplog.records)
