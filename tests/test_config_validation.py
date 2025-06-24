import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.service.service import ChatbotService


def test_start_training_invalid_config() -> None:
    svc = ChatbotService()
    svc._config["num_epochs"] = None
    res = svc.start_training("finetune")
    assert not res["success"]
