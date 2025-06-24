import pytest
pytest.importorskip("torch")
import logging
from pathlib import Path
from src.service.core import ChatbotService
from src.training import train
from src.config import Config

pytestmark = pytest.mark.requires_torch


def test_log_singleline(caplog, tmp_path, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    cfg = Config(num_epochs=2, batch_size=2)
    train(Path('datas/qa_test_10.json'), cfg, model_path=tmp_path/'m.pth')
    msgs = [r.message for r in caplog.records if 'epoch' in r.message]
    assert len(msgs) == 2
    assert msgs[0].startswith('epoch 1/2') and msgs[1].startswith('epoch 2/2')
