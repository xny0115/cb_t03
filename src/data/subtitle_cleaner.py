from __future__ import annotations

import re
from pathlib import Path
from typing import List

try:  # optional dependency
    from charset_normalizer import from_bytes
except Exception:  # pragma: no cover - optional
    from_bytes = None  # type: ignore

_RE_TAG = re.compile(r"<[^>]+>")
_RE_HTML = re.compile(r"&[^;]+;")
_RE_KR = re.compile(r"[\uac00-\ud7a3]")
_ALLOWED_PUNCT = {".", ",", "!", "?"}


def _decode(data: bytes) -> str:
    """Decode bytes with common Korean encodings."""
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    if from_bytes is not None:  # pragma: no cover - best effort
        try:
            best = from_bytes(data).best()
            if best:
                return str(best)
        except Exception:
            pass
    return data.decode("utf-8", "ignore")


def _clean_line(line: str) -> str:
    """Return cleaned text without html entities and unwanted chars."""
    line = _RE_HTML.sub(" ", line)
    result = []
    for ch in line:
        if ch.isalnum() or ch.isspace() or ch in _ALLOWED_PUNCT:
            result.append(ch)
    cleaned = "".join(result)
    return " ".join(cleaned.split()).strip()


def extract_lines(path: Path) -> List[str]:
    """Return list of Korean lines from subtitle file."""
    text = _decode(path.read_bytes())
    text = text.replace("<br>", "\n").replace("&nbsp;", " ")
    text = _RE_TAG.sub("", text)
    lines: List[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if path.suffix.lower() == ".srt":
            if ln.isdigit() or "-->" in ln:
                continue
        if _RE_KR.search(ln):
            ln = _clean_line(ln)
            if ln:
                lines.append(ln)
    return lines


def clean_subtitle_files(root_dir: Path, dst_dir: Path) -> None:
    """Extract lines from all subtitles under ``root_dir`` to ``dst_dir``."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = list(root_dir.glob("*.smi")) + list(root_dir.glob("*.srt"))
    for fp in files:
        lines = extract_lines(fp)
        if not lines:
            continue
        seen: set[str] = set()
        unique: List[str] = []
        for ln in lines:
            if ln not in seen:
                seen.add(ln)
                unique.append(ln)
        out = dst_dir / f"{fp.stem}.txt"
        with open(out, "w", encoding="utf-8") as f:
            for ln in unique:
                f.write(ln + "\n")


__all__ = ["extract_lines", "clean_subtitle_files"]
