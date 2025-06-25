import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.service.service import ChatbotService
from src.utils.validator import REQUIRED_KEYS


def test_auto_tune_updates_config(tmp_path: Path) -> None:
    svc = ChatbotService()
    before = svc.get_config().copy()
    res = svc.auto_tune()
    cfg = res["data"]
    after = svc.get_config()
    for k, v in cfg.items():
        assert after[k] == v
    for k in REQUIRED_KEYS:
        assert k in cfg and cfg[k] is not None
    assert res["success"] and cfg

