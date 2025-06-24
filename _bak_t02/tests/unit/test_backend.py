import pytest
pytest.importorskip("torch")
from src.ui.backend import WebBackend
from src.config import load_config
from src.service import ChatbotService
import time

pytestmark = pytest.mark.requires_torch


def test_backend_cycle(tmp_path):
    backend = WebBackend(ChatbotService())
    cfg = load_config()
    cfg['epochs'] = 1
    backend.set_config(cfg)
    res = backend.start_training()
    assert res['success']
    while True:
        st = backend.get_status()
        msg = st['data']['status_msg']
        if msg in ('done', '') or msg.startswith('error'):
            break
        time.sleep(0.1)
    del_res = backend.delete_model()
    assert del_res["success"]
    inf = backend.infer('hi')
    assert not inf['success']
