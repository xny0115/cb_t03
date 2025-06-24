from __future__ import annotations

"""Morphological analysis utilities using KoNLPy Komoran.

This module normalizes text with ``soynlp`` before analysis and
applies simple post-processing to remove punctuation tokens.
"""

import logging
from typing import List, Dict

from soynlp.normalizer import emoticon_normalize, repeat_normalize
from soynlp.tokenizer import RegexTokenizer
import os
import contextlib
import sys

# avoid noisy JPype warnings on modern JVMs
os.environ.setdefault("JAVA_TOOL_OPTIONS", "--enable-native-access=ALL-UNNAMED")

from konlpy.tag import Komoran

_komoran = None


def get_komoran() -> Komoran | None:
    """Return singleton ``Komoran`` instance."""
    global _komoran
    if _komoran is None:
        try:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull), contextlib.redirect_stdout(devnull):
                _komoran = Komoran()
        except Exception as exc:  # pragma: no cover - best effort
            logging.getLogger(__name__).error("Komoran init failed: %s", exc)
            _komoran = None
    return _komoran

_regex_tokenizer = RegexTokenizer()


def _normalize(text: str) -> str:
    """Return text normalized with ``soynlp`` utilities."""
    text = emoticon_normalize(text, num_repeats=2)
    return repeat_normalize(text, num_repeats=2)


def _postprocess(tokens: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove punctuation tokens."""
    skip = {"SP", "SSO", "SSC", "SC", "SE", "SF", "SY"}
    return [t for t in tokens if t.get("pos") not in skip]


def analyze(text: str) -> List[Dict[str, str]]:
    """Return list of morphological tokens for ``text``."""
    if not text:
        return []
    text = _normalize(text)
    parts: List[tuple[str, str]]
    komoran = get_komoran()
    if komoran is None:
        logging.getLogger(__name__).warning("komoran fallback active")
        parts = [(t, "UNK") for t in _regex_tokenizer.tokenize(text, flatten=True)]
    else:
        try:
            parts = komoran.pos(text)
        except Exception as exc:  # pragma: no cover - best effort
            logging.getLogger(__name__).warning("komoran error: %s", exc)
            parts = [(t, "UNK") for t in _regex_tokenizer.tokenize(text, flatten=True)]

    tokens: List[Dict[str, str]] = [
        {"text": w, "lemma": w, "pos": p} for w, p in parts
    ]
    return _postprocess(tokens)


__all__ = ["analyze"]
