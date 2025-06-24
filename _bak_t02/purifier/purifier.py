"""Dataset purification utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from src.data.morph import analyze


def _strip_quotes(text: str) -> str:
    """Return ``text`` without single or double quotes."""
    return text.replace('"', "").replace("'", "")


@dataclass
class RawPair:
    """Simple raw question-answer pair."""

    question: str
    answer: str


@dataclass
class CleanPair:
    """Cleaned dataset pair with tokens and metadata."""

    question: Dict[str, object]
    answer: Dict[str, object]
    concepts: List[str]
    domain: str


def _extract_concepts(tokens_q: List[Dict[str, str]], tokens_a: List[Dict[str, str]]) -> List[str]:
    """Return unique nouns found in question and answer tokens."""
    nouns = {t["lemma"] for t in tokens_q + tokens_a if t.get("pos", "").startswith("NN")}
    return sorted(nouns)


def _parse_raw(path: Path) -> List[RawPair]:
    """Load raw pairs from ``path`` which can be json or txt."""
    pairs: List[RawPair] = []
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            q = _strip_quotes(str(item.get("question", "")).strip())
            a = _strip_quotes(str(item.get("answer", "")).strip())
            if q and a:
                pairs.append(RawPair(q, a))
    elif path.suffix.lower() == ".txt":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "question" in line and "answer" in line:
                    try:
                        q_part, a_part = line.split("answer", 1)
                        q = _strip_quotes(q_part.split("question", 1)[1].strip(" :,"))
                        a = _strip_quotes(a_part.strip(" :,"))
                        pairs.append(RawPair(q, a))
                    except Exception:
                        continue
    return pairs


def _sanitize_name(name: str) -> str:
    """Return ``name`` stripped of surrounding quotes."""
    return name.strip('"').strip("'")


def _domain_from_name(name: str) -> str:
    """Infer domain string from file name."""
    parts = _sanitize_name(name).split("_")
    if len(parts) > 1:
        return parts[1]
    return parts[0]


def clean_file(src: Path, dst_dir: Path) -> Path:
    """Convert ``src`` raw dataset file to cleaned JSON under ``dst_dir``."""
    raw_pairs = _parse_raw(src)
    cleaned: List[CleanPair] = []
    base = _sanitize_name(src.stem)
    domain = _domain_from_name(base)
    for pair in raw_pairs:
        tok_q = [
            {**t, "text": _strip_quotes(t.get("text", "")), "lemma": _strip_quotes(t.get("lemma", ""))}
            for t in analyze(pair.question)
            if _strip_quotes(t.get("text", ""))
        ]
        tok_a = [
            {**t, "text": _strip_quotes(t.get("text", "")), "lemma": _strip_quotes(t.get("lemma", ""))}
            for t in analyze(pair.answer)
            if _strip_quotes(t.get("text", ""))
        ]
        if len(tok_q) < 1 or len(tok_a) < 1:
            continue
        concepts = _extract_concepts(tok_q, tok_a)
        if len(concepts) < 1:
            continue
        cleaned.append(
            CleanPair(
                question={"text": pair.question, "tokens": tok_q, "concepts": concepts, "domain": domain},
                answer={"text": pair.answer, "tokens": tok_a},
                concepts=concepts,
                domain=domain,
            )
        )
    dst_path = dst_dir / f"{base}.json"
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in cleaned], f, ensure_ascii=False, indent=2)
    _write_log(dst_dir / f"{base}.txt", raw_pairs)
    return dst_path


def _write_log(path: Path, pairs: List[RawPair]) -> None:
    """Write summary log file for ``pairs``."""
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(f"question : {p.question} , answer : {p.answer}\n")
        f.write(f"- 총 질문답 {len(pairs)}개\n")


__all__ = ["clean_file"]
