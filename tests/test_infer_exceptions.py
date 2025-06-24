import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.service.service import ChatbotService
from src.model.dummy import DummyModel


def _setup_service() -> ChatbotService:
    svc = ChatbotService()
    svc.model = DummyModel({"다음 질문에 답하세요. 안녕": "안녕!"})
    return svc


def test_empty_input():
    svc = _setup_service()
    res = svc.infer("")
    assert not res["success"] and res["msg"] == "empty_input"


def test_long_input():
    svc = _setup_service()
    res = svc.infer("a" * (svc.MAX_INPUT_LEN + 1))
    assert not res["success"] and res["msg"] == "too_long"


def test_special_input():
    svc = _setup_service()
    res = svc.infer("😀")
    assert res["msg"] == "no_answer"


def test_unknown_input():
    svc = _setup_service()
    res = svc.infer("모르는 질문")
    assert res["msg"] == "no_answer"


