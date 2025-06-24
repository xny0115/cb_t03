import pytest
pytest.importorskip("torch")
import logging
from pathlib import Path
from src.service.core import ChatbotService
from src.training import train
from src.config import Config

pytestmark = pytest.mark.requires_torch


def test_log_reduction(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    cfg = Config(num_epochs=1, batch_size=2)
    train(Path('datas/qa_test_10.json'), cfg, model_path=tmp_path/'m.pth')
    msgs = [r.message for r in caplog.records]
    assert any('epoch 1/1' in m for m in msgs)
    assert not any('step' in m for m in msgs)
