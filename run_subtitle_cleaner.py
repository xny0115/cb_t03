"""CLI for subtitle dataset cleaner."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.subtitle_cleaner import clean_subtitle_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Subtitle dataset cleaner")
    parser.add_argument(
        "src",
        type=Path,
        nargs="?",
        default=Path("."),
        help="directory containing subtitle files",
    )
    parser.add_argument(
        "dst",
        type=Path,
        nargs="?",
        default=Path("datas/01_pretrain"),
        help="output directory for cleaned text",
    )
    args = parser.parse_args()
    clean_subtitle_files(args.src, args.dst)


if __name__ == "__main__":
    main()
