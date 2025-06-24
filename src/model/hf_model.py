from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import pipeline


@dataclass
class HFModel:
    """간단한 HuggingFace 모델 래퍼."""

    name: str

    def __post_init__(self) -> None:
        cache_dir = Path("models/hf")
        self.pipe = pipeline(
            "text-generation",
            model=self.name,
            device=0 if torch.cuda.is_available() else -1,
            cache_dir=cache_dir,
        )

    def predict(self, instruction: str, inp: str) -> str:
        prompt = f"{instruction.strip()} {inp.strip()}".strip()
        outputs = self.pipe(prompt, max_new_tokens=50)
        if isinstance(outputs, list) and outputs:
            return outputs[0].get("generated_text", "").strip()
        return str(outputs)
