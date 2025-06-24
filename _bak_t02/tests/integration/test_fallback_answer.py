import pytest
pytest.importorskip("torch")
import torch
from src.service.core import ChatbotService
from src.utils.vocab import Tokenizer


class DummyModel:
    def generate(self, *args, **kwargs):
        return torch.tensor([[1]])

pytestmark = pytest.mark.requires_torch


def test_fallback_answer():
    svc = ChatbotService()
    svc.model_exists = True
    svc._tokenizer = Tokenizer({"<pad>":0, "<eos>":1})
    svc._model = DummyModel()
    res = svc.infer("안녕?")
    assert res["success"] and res["data"] == ""
