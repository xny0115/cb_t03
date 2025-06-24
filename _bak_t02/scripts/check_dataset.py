from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.data.morph import analyze

REQUIRED_KEYS = {"text", "lemma", "pos"}
POS_PREFIX = ("N", "V", "XR")


def _load(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _check_tokens(tokens: List[dict], idx: int, part: str) -> int:
    errors = 0
    for i, tok in enumerate(tokens):
        if not REQUIRED_KEYS <= tok.keys():
            print(f"{idx}:{part}:{i} token missing fields")
            errors += 1
            continue
        if not tok["pos"].startswith(POS_PREFIX):
            print(f"{idx}:{part}:{i} invalid pos {tok['pos']}")
            errors += 1
    return errors


def _check_concepts(item: dict, idx: int) -> int:
    q_cons = item.get("question", {}).get("concepts", [])
    a_cons = item.get("answer", {}).get("concepts", [])
    pair = sorted(set(q_cons + a_cons))
    declared = item.get("concepts", [])
    errors = 0
    if declared != pair:
        print(f"concepts mismatch at {idx}")
        errors += 1
    if declared != sorted(set(declared)):
        print(f"concepts not unique/sorted at {idx}")
        errors += 1
    return errors


def check_file(path: Path) -> int:
    data = _load(path)
    errors = 0
    for idx, item in enumerate(data):
        q = item.get("question", {})
        a = item.get("answer", {})
        if "domain" not in q or "concepts" not in q:
            print(f"question fields missing at {idx}")
            errors += 1
        if q.get("domain") != item.get("domain"):
            print(f"domain mismatch at {idx}")
            errors += 1
        errors += _check_tokens(q.get("tokens", []), idx, "q")
        errors += _check_tokens(a.get("tokens", []), idx, "a")
        if analyze(q.get("text", "")) != q.get("tokens"):
            print(f"token mismatch question {idx}")
            errors += 1
        if analyze(a.get("text", "")) != a.get("tokens"):
            print(f"token mismatch answer {idx}")
            errors += 1
        errors += _check_concepts(item, idx)
    return errors


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, nargs="?", default=Path("datas"))
    args = p.parse_args()

    paths = list(args.path.glob("*.json")) if args.path.is_dir() else [args.path]
    total = 0
    for fp in paths:
        total += check_file(fp)
    if total:
        print(f"found {total} issues")
    else:
        print("all checks passed")
