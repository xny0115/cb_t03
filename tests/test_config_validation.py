import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.service.service import ChatbotService


def test_start_training_invalid_config() -> None:
    svc = ChatbotService()
    svc._config["num_epochs"] = None
    from src.service import service as svc_module
    orig_read_ini = svc_module._read_ini
    try:
        svc_module._read_ini = lambda path="trainconfig.ini": {}
        res = svc.start_training("finetune")
    finally:
        svc_module._read_ini = orig_read_ini
    assert not res["success"]
