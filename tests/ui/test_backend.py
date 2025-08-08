import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ui.backend import WebBackend
from src.service.service import ChatbotService


def test_get_config_returns_success() -> None:
    """`/get_config` 호출 시 성공 여부와 설정 딕셔너리를 반환해야 한다."""
    svc = MagicMock(spec=ChatbotService)
    svc.get_config.return_value = {"model_path": "m.pth", "device": "cuda"}
    backend = WebBackend(svc)
    res = backend.get_config()
    assert res == {"success": True, "data": {"model_path": "m.pth", "device": "cuda"}}
