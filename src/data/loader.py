from __future__ import annotations

"""Dataset loading helpers.

폴더 내 전체 데이터셋 자동 통합 구조를 지원한다.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Set

import logging
import json


@dataclass
class InstructionSample:
    instruction: str
    input: str
    output: str


def _collect_files(path: Path, patterns: Tuple[str, ...]) -> List[Path]:
    """Return sorted list of files matching patterns inside path."""
    if path.is_file():
        return [path]
    files: List[Path] = []
    for p in patterns:
        files.extend(sorted(path.rglob(p)))
    return files


def _load_jsonl(fp: Path) -> List[dict]:
    """Load json objects from a jsonl file."""
    logger = logging.getLogger(__name__)
    items: List[dict] = []
    try:
        with open(fp, encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as exc:  # pragma: no cover - warn only
                    logger.warning("invalid json in %s:%d - %s", fp, idx, exc)
                    continue
                items.append(obj)
    except Exception as exc:  # pragma: no cover - warn only
        logger.warning("failed to read %s: %s", fp, exc)
    return items


def _load_text(fp: Path) -> List[str]:
    """Load lines from a text file."""
    logger = logging.getLogger(__name__)
    lines: List[str] = []
    try:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    except Exception as exc:  # pragma: no cover - warn only
        logger.warning("failed to read %s: %s", fp, exc)
    return lines


def load_instruction_dataset(path: Path) -> List[InstructionSample]:
    """Merge all jsonl files under ``path`` into one dataset."""
    files = _collect_files(path, ("*.jsonl",))
    samples: List[InstructionSample] = []
    seen: Set[Tuple[str, str, str]] = set()
    for fp in files:
        for item in _load_jsonl(fp):
            if not isinstance(item, dict):
                continue
            ins = item.get("instruction", "").strip()
            inp = item.get("input", "")
            out = item.get("output", "")
            if not out:
                continue
            key = (ins, inp, out)
            if key in seen:
                continue
            seen.add(key)
            samples.append(InstructionSample(ins, inp, out))
    return samples


def load_pretrain_dataset(path: Path) -> List[str]:
    """Merge all txt files under ``path`` into one dataset."""
    files = _collect_files(path, ("*.txt",))
    lines: List[str] = []
    seen: Set[str] = set()
    for fp in files:
        for line in _load_text(fp):
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
    return lines


def get_text_stats(path: Path) -> dict[str, float | int]:
    """Return statistics about txt dataset under ``path``."""
    files = _collect_files(path, ("*.txt",))
    file_count = len(files)
    total_lines = 0
    empty_lines = 0
    dup_lines = 0
    total_chars = 0
    max_chars = 0
    min_chars = float("inf")
    seen: Set[str] = set()

    for fp in files:
        for raw in open(fp, encoding="utf-8"):
            total_lines += 1
            line = raw.strip()
            if not line:
                empty_lines += 1
                continue
            if line in seen:
                dup_lines += 1
            else:
                seen.add(line)
            ln = len(line)
            total_chars += ln
            max_chars = max(max_chars, ln)
            if ln < min_chars:
                min_chars = ln

    non_empty = total_lines - empty_lines
    avg_chars = total_chars / non_empty if non_empty else 0
    dup_ratio = dup_lines / non_empty if non_empty else 0
    if min_chars == float("inf"):
        min_chars = 0

    return {
        "files": file_count,
        "lines": total_lines,
        "empty_lines": empty_lines,
        "dup_lines": dup_lines,
        "dup_ratio": dup_ratio,
        "avg_chars": avg_chars,
        "max_chars": max_chars,
        "min_chars": min_chars,
    }


def get_dataset_info(
    pre_dir: Path, ft_dir: Path, add_dir: Path
) -> tuple[int, int, int, int, list[str]]:
    """Return dataset size and token count across all directories."""
    txt_lines = 0
    json_lines = 0
    tokens = 0
    skipped: list[str] = []

    for path in [pre_dir, ft_dir, add_dir]:
        if path.name.startswith("01"):
            continue

        for fp in path.glob("*.txt"):
            lines = _load_text(fp)
            txt_lines += len(lines)
            tokens += sum(len(l) for l in lines)

        for fp in path.glob("*.jsonl"):
            for item in _load_jsonl(fp):
                if not isinstance(item, dict) or "instruction" not in item:
                    continue
                ins = item.get("instruction", "")
                inp = item.get("input", "")
                out = item.get("output", "")
                json_lines += 1
                tokens += len(ins) + len(inp) + len(out)

        for fp in path.glob("*.json"):
            skipped.append(fp.name)

    total = txt_lines + json_lines
    return total, tokens, txt_lines, json_lines, skipped


def load_dataset(path: Path) -> List[InstructionSample]:
    """이전 호환을 위해 유지."""
    return load_instruction_dataset(path)


__all__ = [
    "InstructionSample",
    "load_instruction_dataset",
    "load_pretrain_dataset",
    "get_text_stats",
    "load_dataset",
    "get_dataset_info",
]
