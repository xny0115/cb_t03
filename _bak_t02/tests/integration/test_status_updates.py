import pytest
pytest.importorskip("torch")
import time
from src.service.core import ChatbotService

pytestmark = pytest.mark.requires_torch

def test_status_updates(tmp_path):
    svc = ChatbotService()
    svc.update_config({'epochs': 1})
    svc.start_training()
    seen = False
    for _ in range(20):
        st = svc.get_status()["data"]
        if 0 < st.get("progress", 0) <= 1:
            seen = True
            break
        time.sleep(0.1)
    svc.stop_training()
    assert seen

