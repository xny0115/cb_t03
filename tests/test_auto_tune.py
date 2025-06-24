import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.service.service import ChatbotService


def test_auto_tune_updates_config(tmp_path: Path) -> None:
    svc = ChatbotService()
    before = svc.get_config().copy()
    cfg = svc.auto_tune()
    after = svc.get_config()
    for k, v in cfg.items():
        assert after[k] == v
    assert cfg  # ensure not empty

