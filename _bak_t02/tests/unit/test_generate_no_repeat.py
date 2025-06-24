import pytest
pytest.importorskip("torch")
import torch
from src.model.transformer import Seq2SeqTransformer

pytestmark = pytest.mark.requires_torch

def test_no_repeat():
    model = Seq2SeqTransformer(vocab_size=5)
    def forward_stub(self, src, tgt):
        out = torch.zeros(tgt.size(0), src.size(1), self.fc_out.out_features)
        out[-1, :, 2] = 10.0
        return out
    model.forward = forward_stub.__get__(model, Seq2SeqTransformer)
    src = torch.tensor([[0]])
    out = model.generate(src, max_new_tokens=5, no_repeat_ngram=2)
    ids = out.view(-1).tolist()
    bigrams = [tuple(ids[i:i+2]) for i in range(len(ids)-1)]
    assert len(bigrams) == len(set(bigrams))
