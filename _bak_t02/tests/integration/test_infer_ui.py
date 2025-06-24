import pytest
pytest.importorskip("torch")
from src.ui.backend import WebBackend
from src.service import ChatbotService
import time

pytestmark = pytest.mark.requires_torch


def test_infer_ui(tmp_path):
    backend = WebBackend(ChatbotService())
    backend.set_config({'epochs': 1})
    backend.start_training()
    for _ in range(30):
        if backend.get_status()['data']['status_msg'] == 'done':
            break
        time.sleep(0.1)
    res = backend.infer('hello')
    assert res['success'] and len(res['data']) >= 5
