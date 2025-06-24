"""CLI for dataset purifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from purifier import clean_file, subtitle_to_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset purifier")
    parser.add_argument(
        "src",
        type=Path,
        nargs="?",
        default=Path("datas_raw"),
        help="raw dataset file or directory",
    )
    parser.add_argument("dst", type=Path, nargs="?", default=Path("datas"), help="output directory")
    args = parser.parse_args()

    if args.src.is_dir():
        files = sorted(args.src.glob("*"))
    else:
        files = [args.src]

    args.dst.mkdir(parents=True, exist_ok=True)

    for fp in files:
        if not fp.is_file():
            continue
        ext = fp.suffix.lower()
        if ext in {".json", ".txt"}:
            out = clean_file(fp, args.dst)
            print(f"saved {out}")
        elif ext in {".srt", ".smi"}:
            out = subtitle_to_json(fp, args.dst)
            print(f"saved {out}")


if __name__ == "__main__":
    main()
