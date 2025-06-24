from pathlib import Path
from src.utils.persist import load_json, save_json

def test_persist_roundtrip(tmp_path):
    path = tmp_path / "data.json"
    data = {"a": 1}
    save_json(data, path)
    assert load_json(path) == data
