import pytest
pytest.importorskip("torch")
import time
from src.service.core import ChatbotService

pytestmark = pytest.mark.requires_torch


def test_chat_flow(tmp_path):
    svc = ChatbotService()
    svc.update_config({'epochs': 1})
    svc.start_training()
    for _ in range(30):
        st = svc.get_status()["data"]["status_msg"]
        if st.startswith("error") or st == "done":
            break
        time.sleep(0.1)
    res = svc.infer("인공지능이란 뭐야?")
    assert res["success"]
    assert isinstance(res["data"], str)
    assert svc.delete_model() is True
