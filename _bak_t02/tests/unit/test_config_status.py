import pytest
pytest.importorskip("torch")
from src.service.core import ChatbotService
from pathlib import Path
import json

pytestmark = pytest.mark.requires_torch

def test_update_and_status(tmp_path):
    svc = ChatbotService()
    cfg = {"epochs": 3, "batch_size": 2, "lr": 0.001}
    ok, msg = svc.update_config(cfg)
    assert ok and msg == "saved"
    saved = json.load(open(Path("configs/current.json")))
    assert saved["epochs"] == 3
    status = svc.get_status()
    assert status["success"] and "cpu_usage" in status["data"]
