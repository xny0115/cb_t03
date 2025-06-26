from pathlib import Path

from src.data.loader import load_pretrain_dataset


def test_pretrain_filter(tmp_path: Path) -> None:
    pre = tmp_path / "pretrain"
    pre.mkdir()
    (pre / "a.txt").write_text("a\nb\ncd\n", encoding="utf-8")
    lines = load_pretrain_dataset(pre)
    assert lines == ["cd"]

