from __future__ import annotations

"""Dataset normalization utilities."""
from pathlib import Path
import json
from typing import Dict

from .morph import analyze
from .preprocess import extract_concepts, infer_domain


def normalize_file(path: Path) -> int:
    """Normalize dataset JSON file in place and return change count."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    changed = 0
    for item in data:
        q = item.get("question", {})
        a = item.get("answer", {})
        q_tokens = analyze(q.get("text", ""))
        a_tokens = analyze(a.get("text", ""))
        if q.get("tokens") != q_tokens:
            q["tokens"] = q_tokens
            changed += 1
        if a.get("tokens") != a_tokens:
            a["tokens"] = a_tokens
            changed += 1
        conc = extract_concepts(q_tokens + a_tokens)
        if item.get("concepts") != conc:
            item["concepts"] = conc
            changed += 1
        dom = infer_domain(q_tokens + a_tokens)
        if item.get("domain") != dom:
            item["domain"] = dom
            changed += 1
        q["concepts"] = conc
        q["domain"] = dom
        item["question"] = q
        item["answer"] = a
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return changed


def normalize_path(path: Path) -> Dict[str, int]:
    """Normalize file or directory and return per-file changes."""
    files = [path] if path.is_file() else sorted(path.glob("*.json"))
    result: Dict[str, int] = {}
    for fp in files:
        result[fp.name] = normalize_file(fp)
    return result


__all__ = ["normalize_file", "normalize_path"]
