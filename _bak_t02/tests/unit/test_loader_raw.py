import pytest
pytest.importorskip("soynlp")
import json
from pathlib import Path
from src.data import QADataset

pytestmark = pytest.mark.requires_torch


def test_dataset_loading_raw(tmp_path: Path) -> None:
    file = tmp_path / "raw.json"
    json.dump(["문장"], open(file, "w", encoding="utf-8"), ensure_ascii=False)
    ds = QADataset(file)
    assert len(ds) == 1
    q, a = ds[0]
    assert q == ""
    assert a == "문장"
