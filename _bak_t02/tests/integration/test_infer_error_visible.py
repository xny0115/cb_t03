import pytest
pytest.importorskip("torch")
from src.ui.backend import WebBackend
from src.service import ChatbotService

pytestmark = pytest.mark.requires_torch

def test_infer_error_visible():
    backend = WebBackend(ChatbotService())
    captured = {}
    def show(msg):
        captured['msg'] = msg
    # mimic onSend logic
    res = backend.infer('hello')
    if not res['success']:
        show(res['msg'])
    assert 'model_missing' in captured['msg']
