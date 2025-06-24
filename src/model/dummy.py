from __future__ import annotations

import torch
from pathlib import Path


class DummyModel:
    """매우 단순한 매핑 기반 모델."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def predict(self, instruction: str, inp: str) -> str:
        return self.mapping.get(inp.strip(), "")


def save_model(model: DummyModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"mapping": model.mapping, "pad": "0" * 1_048_576}
    torch.save(data, path)
    if not path.exists() or path.stat().st_size < 1_000_000:
        raise RuntimeError("모델 저장 실패: 생성 실패 또는 용량 미달")


def load_model(path: Path) -> DummyModel:
    data = torch.load(path, map_location="cpu")
    mapping = data.get("mapping", {})
    return DummyModel(mapping)
