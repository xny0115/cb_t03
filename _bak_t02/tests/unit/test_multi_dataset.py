import pytest
pytest.importorskip("soynlp")
from pathlib import Path
import json
from src.data.loader import load_all

pytestmark = pytest.mark.requires_torch


def test_multi_dataset(tmp_path):
    data_dir = tmp_path / "datas"
    data_dir.mkdir()
    sub = data_dir / "sub"
    sub.mkdir()
    counts = [1, 2, 3]
    for idx, cnt in enumerate(counts):
        fp = (sub if idx == 2 else data_dir) / f"{idx}.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "question": {"text": f"q{idx}_{j}"},
                        "answer": {"text": "a"},
                    }
                    for j in range(cnt)
                ],
                f,
            )
    samples = load_all(data_dir)
    assert len(samples) == sum(counts)

