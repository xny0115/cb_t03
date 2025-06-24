import pytest
pytest.importorskip("soynlp")
import json
from pathlib import Path
from src.data.loader import QADataset
from src.utils.vocab import build_vocab

pytestmark = pytest.mark.requires_torch


def test_vocab_includes_endings(tmp_path: Path) -> None:
    data = [
        {
            "question": {
                "text": "안녕",
                "tokens": [{"text": "안녕", "lemma": "안녕", "pos": "NNG"}],
            },
            "answer": {
                "text": "반가워",
                "tokens": [
                    {"text": "반갑", "lemma": "반갑", "pos": "VA"},
                    {"text": "어", "lemma": "어", "pos": "EF"},
                ],
            },
            "concepts": ["인사"],
            "domain": ""
        }
    ]
    file_path = tmp_path / "sample.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    ds = QADataset(file_path)
    vocab = build_vocab(ds)
    assert "어" in vocab
