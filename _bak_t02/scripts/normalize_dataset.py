from __future__ import annotations

"""CLI for dataset normalization."""
import argparse
from pathlib import Path
from src.data.normalize import normalize_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize dataset")
    parser.add_argument("path", type=Path, nargs="?", default=Path("datas"))
    args = parser.parse_args()
    result = normalize_path(args.path)
    for name, cnt in result.items():
        print(f"{name}: {cnt} changes")


if __name__ == "__main__":
    main()
