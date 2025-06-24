import pytest
pytest.importorskip("torch")
from src.ui.backend import WebBackend
from src.service import ChatbotService
from src.utils.logger import setup_logger, LOG_PATH
import time

pytestmark = pytest.mark.requires_torch


def test_epoch_apply(tmp_path):
    setup_logger()
    backend = WebBackend(ChatbotService())
    backend.set_config({'epochs': 7})
    backend.start_training()
    time.sleep(2)
    text = LOG_PATH.read_text(encoding='utf-8')
    backend.delete_model()
    assert 'training epochs=7' in text
