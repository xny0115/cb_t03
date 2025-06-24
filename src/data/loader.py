from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class InstructionSample:
    instruction: str
    input: str
    output: str


def load_dataset(path: Path) -> List[InstructionSample]:
    """JSON 파일에서 Instruction 샘플을 로드한다."""
    files = [path] if path.is_file() else sorted(path.glob("*.json"))
    samples: List[InstructionSample] = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        for item in data:
            if not isinstance(item, dict):
                continue
            ins = item.get("instruction", "").strip()
            inp = item.get("input", "")
            out = item.get("output", "")
            if not out:
                continue
            samples.append(InstructionSample(ins, inp, out))
    return samples
