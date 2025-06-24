import re
import json
from pathlib import Path
from typing import List

try:
    from charset_normalizer import from_bytes
except Exception:  # pragma: no cover - optional dependency
    from_bytes = None  # type: ignore

_RE_TAG = re.compile(r"<[^>]+>")
_RE_KR = re.compile(r"[\uac00-\ud7a3]")
_ALLOWED_PUNCT = {".", ",", "!", "?"}


def _clean_line(line: str) -> str:
    """Return ``line`` without disallowed punctuation."""
    result = []
    for ch in line:
        if ch.isalnum() or ch.isspace() or ch in _ALLOWED_PUNCT:
            result.append(ch)
    cleaned = "".join(result)
    return " ".join(cleaned.split()).strip()


def _decode(data: bytes) -> str:
    """Decode bytes with common Korean encodings and fallback detection."""
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    if from_bytes is not None:  # pragma: no cover - best effort
        try:
            result = from_bytes(data).best()
            if result:
                return str(result)
        except Exception:
            pass
    return data.decode("utf-8", "ignore")


def extract_lines(path: Path) -> List[str]:
    """Return list of Korean lines from subtitle file."""
    text = _decode(path.read_bytes())
    text = text.replace("<br>", "\n").replace("&nbsp;", " ")
    text = _RE_TAG.sub("", text)
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned: List[str] = []
    for ln in lines:
        if ln and _RE_KR.search(ln):
            ln = _clean_line(ln)
            if ln:
                cleaned.append(ln)
    return cleaned


def subtitle_to_json(src: Path, dst_dir: Path) -> Path:
    """Write extracted lines from ``src`` to JSON in ``dst_dir``."""
    lines = extract_lines(src)
    dst = dst_dir / f"{src.stem}.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=2)
    return dst


__all__ = ["extract_lines", "subtitle_to_json"]
