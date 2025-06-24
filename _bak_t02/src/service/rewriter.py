from __future__ import annotations

"""Simple rewriting utilities based on existing dataset phrases."""

import random
import re
from pathlib import Path
from typing import List, Set

from ..data.loader import QADataset
from ..data.morph import analyze


def _extract_keywords(text: str) -> Set[str]:
    """Return set of lemma for nouns and verbs."""
    tokens = analyze(text)
    keywords = {
        t["lemma"] for t in tokens if t.get("pos", "").startswith(("NN", "VV"))
    }
    return keywords


def _find_reference(text: str, ds: QADataset) -> str:
    """Return answer text with most shared keywords."""
    kws = _extract_keywords(text)
    best = ""
    best_score = -1
    for pair in ds.pairs:
        pair_kws = {
            t["lemma"]
            for t in pair.tokens_q + pair.tokens_a
            if t.get("pos", "").startswith(("NN", "VV"))
        }
        score = len(kws & pair_kws)
        if score > best_score:
            best_score = score
            best = pair.answer
    return best or (ds.pairs[0].answer if ds.pairs else "")


def _shuffle_clauses(text: str) -> str:
    """Return text with clauses reordered for variety."""
    parts = [p.strip() for p in re.split(r"[.,]", text) if p.strip()]
    if len(parts) > 1:
        first = parts.pop(0)
        random.shuffle(parts)
        parts.append(first)
        return ", ".join(parts)
    return text.rstrip(".")


def rewrite(text: str, data_path: Path | None = None) -> str:
    """Rewrite ``text`` referencing dataset sentences."""
    ds = QADataset(data_path or Path("datas"))
    ref = _find_reference(text, ds)
    core = _shuffle_clauses(ref)
    endings = ["야", "이지", "라고 알려져 있어", "라고 해"]
    return f"{core.strip()} {random.choice(endings)}"


__all__ = ["rewrite"]
