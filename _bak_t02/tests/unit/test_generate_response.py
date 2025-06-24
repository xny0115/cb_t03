import torch
from src.generate_utils import generate_response, manual_vocab

class DummyTokenizer:
    def __init__(self):
        self.vocab_size = len(manual_vocab)
    def encode(self, text, return_tensors=None):
        return torch.tensor([[1]])
    def decode(self, ids):
        return ""
    def get_vocab(self):
        return {k: i for i, k in enumerate(manual_vocab.keys())}

class DummyModel:
    def eval(self):
        pass
    def generate(self, src, **kwargs):
        return torch.tensor([[219, 25, 249]])

def test_generate_response():
    tok = DummyTokenizer()
    model = DummyModel()
    out = generate_response(model, tok, "hi", {})
    assert out == "명절갖추바느질"
    assert len(out) >= 5
