import pytest
pytest.importorskip("torch")
import torch
from src.service.core import ChatbotService
from src.utils.vocab import Tokenizer

class DummyModel:
    def generate(self, *args, **kwargs):
        return torch.tensor([[0,2,3,4,2,3,4,1]])

pytestmark = pytest.mark.requires_torch

def test_no_filter():
    svc = ChatbotService()
    svc.model_exists = True
    svc._tokenizer = Tokenizer({"<pad>":0,"<eos>":1,"위해":2,"꼭":3,"써야":4})
    svc._model = DummyModel()
    res = svc.infer("hi")
    assert res["data"] == "위해 꼭 써야 위해 꼭 써야"
