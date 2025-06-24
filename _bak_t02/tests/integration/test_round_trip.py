import pytest
pytest.importorskip("torch")
from src.ui.backend import WebBackend
from src.config import load_config
from src.service import ChatbotService

backend = WebBackend(ChatbotService())

pytestmark = pytest.mark.requires_torch


def test_train_infer_cycle(tmp_path):
    # update config to run 1 epoch for speed
    cfg = load_config()
    cfg['epochs'] = 1
    backend.set_config(cfg)
    backend.start_training()
    # wait for training to finish
    import time
    for _ in range(20):
        status = backend.get_status()['data']['status_msg']
        if status.startswith('error') or status == 'done':
            break
        time.sleep(0.1)
    infer_res = backend.infer('인공지능이란 뭐야?')
    assert infer_res['success']
    assert isinstance(infer_res['data'], str)
    assert backend.delete_model()["success"]
    infer_res2 = backend.infer('인공지능이란 뭐야?')
    assert not infer_res2['success']
