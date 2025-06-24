import pytest
pytest.importorskip("torch")
from src.ui.backend import WebBackend
from src.service import ChatbotService
import time
from pathlib import Path

pytestmark = pytest.mark.requires_torch


def test_ui_backend_flow(tmp_path):
    backend = WebBackend(ChatbotService())
    backend.set_config({'epochs': 1})
    out = backend.start_training()
    assert out["success"]
    time.sleep(3)
    assert backend.get_status()["data"]["progress"] > 0
    assert Path("models/current.pth").exists()
    out2 = backend.delete_model()
    assert out2["success"]
    assert not Path("models/current.pth").exists()
