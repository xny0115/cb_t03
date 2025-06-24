from __future__ import annotations

"""Utility functions for dataset preprocessing."""

from typing import List, Dict, Set

from .morph import analyze

_NOUN_PREFIX = "NN"

# simple domain keywords; expand as needed
_DOMAIN_KEYWORDS = {
    "고기": {"고기", "돼지고기", "삼겹살", "목살", "돼지"},
}


def extract_concepts(tokens: List[Dict[str, str]]) -> List[str]:
    """Return sorted unique noun lemmas from ``tokens``."""
    nouns: Set[str] = {
        t.get("lemma") or t.get("text", "")
        for t in tokens
        if t.get("pos", "").startswith(_NOUN_PREFIX)
    }
    return sorted(nouns)


def infer_domain(tokens: List[Dict[str, str]]) -> str:
    """Return domain string inferred from ``tokens``."""
    lemmas = {t.get("lemma") or t.get("text", "") for t in tokens}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if lemmas & keywords:
            return domain
    return ""


__all__ = ["extract_concepts", "infer_domain", "analyze"]
