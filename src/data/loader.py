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


def _load_jsonl(fp: Path) -> List[dict]:
    items: List[dict] = []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            items.append(obj)
    return items


def load_instruction_dataset(path: Path) -> List[InstructionSample]:
    """jsonl 파일에서 instruction 샘플을 로드한다."""
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    samples: List[InstructionSample] = []
    for fp in files:
        for item in _load_jsonl(fp):
            if not isinstance(item, dict):
                continue
            ins = item.get("instruction", "").strip()
            inp = item.get("input", "")
            out = item.get("output", "")
            if not out:
                continue
            samples.append(InstructionSample(ins, inp, out))
    return samples


def load_pretrain_dataset(path: Path) -> List[str]:
    """텍스트 라인별 사전학습 데이터 로드."""
    files = [path] if path.is_file() else sorted(path.glob("*.txt"))
    lines: List[str] = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    return lines


def load_dataset(path: Path) -> List[InstructionSample]:
    """이전 호환을 위해 유지."""
    return load_instruction_dataset(path)


__all__ = [
    "InstructionSample",
    "load_instruction_dataset",
    "load_pretrain_dataset",
    "load_dataset",
]
