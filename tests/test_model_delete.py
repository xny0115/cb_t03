# CPU 전용 테스트(gpu 없음 환경용)
# 실행 전: ALLOW_CPU_TRAINING=1 설정
# 본 프로젝트 메인 코드는 GPU 전제이며, 여기서는 기능 검증만 수행합니다.

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
    assert (tmp_path / "STOP").exists()
