from __future__ import annotations

"""Token-to-sentence restorer based on basic POS rules.

입력 토큰 배열을 간단한 형태소 규칙으로 다시 문장 형태로 복원한다. ``text`` 필드가
존재하면 우선 사용하고, 없으면 ``lemma`` 필드를 사용한다. 조사는 앞 단어와 붙여 쓰며
어미와 연결어미, 접사류(X 계열)는 모두 이전 단어에 붙여 쓴다. 마지막 토큰이 서술격
어휘인데 종결 어미가 없으면 ``다``를 자동으로 붙여 준다."""

from typing import Dict, List


def _has_batchim(word: str) -> bool:
    """Return True if ``word`` ends with a final consonant."""
    if not word:
        return False
    code = ord(word[-1])
    return 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0


def _choose_particle(word: str, josa_type: str) -> str:
    """Return the particle variant for ``word`` based on ``josa_type``."""
    if josa_type == "JKO":
        return "을" if _has_batchim(word) else "를"
    return "은" if _has_batchim(word) else "는"


REPLACEMENTS = {
    "하아": "해",
    "하어": "해",
    "하였": "했",
    "리어": "려",
    "주ㄹ": "줄",
    "이ㄴ": "인",
}


def _apply_corrections(word: str) -> str:
    """Return post-processed ``word`` applying common contractions."""
    for src, tgt in REPLACEMENTS.items():
        if src in word:
            word = word.replace(src, tgt)
    return word


def restore_sentence(tokens: List[Dict[str, str]]) -> str:
    """Assemble a sentence from morphological tokens."""
    words: List[str] = []
    attachable = {"VCP", "VNP", "VX"}
    n = len(tokens)
    for idx, tok in enumerate(tokens):
        text = tok.get("text") or tok.get("lemma", "")
        pos = tok.get("pos", "")
        if not text:
            continue

        if pos.startswith(("J", "E", "X")) or pos in attachable or pos in {"SF", "SP", "SSO", "SSC", "SE"}:
            if words:
                words[-1] += text
            else:
                words.append(text)
            continue

        if pos.startswith("N"):
            next_pos = tokens[idx + 1].get("pos", "") if idx + 1 < n else ""
            if not next_pos.startswith(("J", "X", "E")):
                particle_type = "JKO" if next_pos.startswith("VV") else "JKS"
                text += _choose_particle(text, particle_type)
        words.append(text)

    if tokens:
        last_pos = tokens[-1].get("pos", "")
        if last_pos in {"VV", "VA", "VX", "VCP", "VCN"} and not words[-1].endswith(("다", "요")):
            words[-1] += "다"

    words = [_apply_corrections(w) for w in words]
    sentence = " ".join(words)
    sentence = sentence.replace(" .", ".").replace(" ,", ",")
    return sentence


__all__ = ["restore_sentence"]
