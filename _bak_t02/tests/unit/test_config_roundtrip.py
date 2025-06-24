import pytest
pytest.importorskip("torch")
from src.ui.backend import WebBackend
from src.service import ChatbotService

pytestmark = pytest.mark.requires_torch


def test_config_roundtrip(tmp_path):
    backend = WebBackend(ChatbotService())
    backend.set_config({'epochs': 3, 'batch_size': 4})
    cfg = backend.get_status()['data']['current_config']
    assert cfg['epochs'] == 3 and cfg['batch_size'] == 4
