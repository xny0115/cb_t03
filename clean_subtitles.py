from __future__ import annotations

"""CLI for subtitle dataset cleaning."""

import argparse
from pathlib import Path

from src.data.subtitle_cleaner import extract_lines, clean_subtitle_files


def _clean_file(src: Path, dst: Path) -> None:
    lines = extract_lines(src)
    if not lines:
        return
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / f"{src.stem}.txt"
    with open(out, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Subtitle cleaner")
    parser.add_argument("src", type=Path, nargs="?", default=Path("."))
    parser.add_argument("dst", type=Path, nargs="?", default=Path("datas/01_pretrain"))
    args = parser.parse_args()

    if args.src.is_file():
        _clean_file(args.src, args.dst)
        return
    clean_subtitle_files(args.src, args.dst)
    print(f"cleaned subtitles from {args.src} to {args.dst}")


if __name__ == "__main__":
    main()
