import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.service.service import ChatbotService


def test_delete_model_all(tmp_path: Path) -> None:
    svc = ChatbotService()
    svc.model_dir = tmp_path
    for name in ["pretrain.pth", "finetune.pth", "additional_finetune.pth"]:
        p = tmp_path / name
        p.write_bytes(b"0" * 1_048_576)
    svc.model_path = tmp_path / "finetune.pth"
    assert svc.delete_model()
    assert list(tmp_path.glob("*.pth")) == []
