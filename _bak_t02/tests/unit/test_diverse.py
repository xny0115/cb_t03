import pytest
pytest.importorskip("torch")
import torch
from src.service.core import ChatbotService
from src.utils.vocab import Tokenizer

class DummyModel:
    def __init__(self):
        self.calls = 0
    def generate(self, *args, **kwargs):
        self.calls += 1
        return torch.tensor([[0,2,1]]) if self.calls == 1 else torch.tensor([[0,3,1]])

pytestmark = pytest.mark.requires_torch

def test_diverse_outputs():
    svc = ChatbotService()
    svc.model_exists = True
    svc._tokenizer = Tokenizer({"<pad>":0,"<eos>":1,"안녕":2,"하이":3})
    svc._model = DummyModel()
    a1 = svc.infer("안녕")['data']
    a2 = svc.infer("안녕")['data']
    assert a1 != a2
