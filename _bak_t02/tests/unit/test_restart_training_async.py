import pytest
pytest.importorskip("torch")
import json
import time
from src.service.core import ChatbotService
from src.utils.logger import setup_logger, LOG_PATH

pytestmark = pytest.mark.requires_torch


def test_restart_training_async(tmp_path):
    setup_logger()
    svc = ChatbotService()
    svc.model_path = tmp_path / "current.pth"
    svc.update_config({"epochs": 1})
    svc.start_training()
    for _ in range(30):
        msg = svc.get_status()["data"]["status_msg"]
        if msg in ("done", "") or msg.startswith("error"):
            break
        time.sleep(0.1)
    meta = svc.model_path.with_suffix(".meta.json")
    assert meta.exists()
    first = json.load(open(meta))
    assert first["epochs_done"] == 1
    svc.start_training()
    for _ in range(30):
        msg = svc.get_status()["data"]["status_msg"]
        if msg in ("done", "") or msg.startswith("error"):
            break
        time.sleep(0.1)
    again = json.load(open(meta))
    assert again["epochs_done"] == 2
    text = LOG_PATH.read_text(encoding="utf-8")
    assert "resume training from epoch 1" in text
