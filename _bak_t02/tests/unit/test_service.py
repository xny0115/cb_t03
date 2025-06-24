import pytest
pytest.importorskip("torch")
import time
from pathlib import Path
from src.service.core import ChatbotService

pytestmark = pytest.mark.requires_torch


def test_start_training_creates_model(tmp_path):
    svc = ChatbotService()
    svc.model_path = tmp_path / "current.pth"
    svc.update_config({"epochs": 1})
    res = svc.start_training()
    assert res["success"]
    for _ in range(30):
        msg = svc.get_status()["data"]["status_msg"]
        if msg == "done" or msg.startswith("error"):
            break
        time.sleep(0.1)
    assert svc.model_path.exists()
    assert svc.model_path.stat().st_size >= 1_000_000
