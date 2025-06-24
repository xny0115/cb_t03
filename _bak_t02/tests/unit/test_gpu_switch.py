import pytest
pytest.importorskip("torch")
import time
from src.service.core import ChatbotService
from src.utils.logger import setup_logger, LOG_PATH

pytestmark = pytest.mark.requires_torch


def test_gpu_switch(monkeypatch):
    setup_logger()
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    svc = ChatbotService()
    svc.update_config({'epochs': 1})
    svc.start_training()
    for _ in range(30):
        msg = svc.get_status()["data"]["status_msg"]
        if msg in ("done", "") or msg.startswith("error"):
            break
        time.sleep(0.1)
    svc.stop_training()
    assert "Device selected: cuda" in LOG_PATH.read_text(encoding="utf-8")
