import pytest
pytest.importorskip("torch")
import time
from src.service.core import ChatbotService
from src.utils.logger import setup_logger

pytestmark = pytest.mark.requires_torch


def test_infer_flow_success(tmp_path):
    setup_logger()
    svc = ChatbotService()
    svc.update_config({'epochs': 1})
    svc.start_training()
    for _ in range(30):
        if svc.get_status()['data']['status_msg'] == 'done':
            break
        time.sleep(0.1)
    res = svc.infer('안녕')
    assert res['success'] and len(res['data']) >= 5
