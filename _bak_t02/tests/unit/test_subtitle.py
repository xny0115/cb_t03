import pytest
pytest.importorskip("soynlp")
from pathlib import Path
from purifier.subtitle import extract_lines, subtitle_to_json

pytestmark = pytest.mark.requires_torch


def test_extract_lines():
    lines = extract_lines(Path("_test.smi"))
    assert any("하비 덴트" in line for line in lines)
    assert all('"' not in ln and "'" not in ln and '-' not in ln for ln in lines)


def test_subtitle_to_json(tmp_path):
    out = subtitle_to_json(Path("_test.smi"), tmp_path)
    assert out.exists()
    data = out.read_text(encoding="utf-8")
    assert "하비" in data
