import pytest
pytest.importorskip("torch")
from src.ui.backend import WebBackend
from src.config import load_config
from src.service import ChatbotService

backend = WebBackend(ChatbotService())

pytestmark = pytest.mark.requires_torch


def test_web_backend_cycle(tmp_path):
    cfg = load_config()
    cfg['epochs'] = 1
    backend.set_config(cfg)
    backend.start_training()
    import time
    for _ in range(30):
        status = backend.get_status()['data']['status_msg']
        if status.startswith('error') or status == 'done':
            break
        time.sleep(0.1)
    result = backend.infer('테스트')
    assert result['success']
    assert isinstance(result['data'], str)
    assert backend.delete_model()["success"]
    result2 = backend.infer('테스트')
    assert not result2['success']
