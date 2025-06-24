import pytest
pytest.importorskip("soynlp")
from src.data.morph import analyze

pytestmark = pytest.mark.requires_torch


def test_morph_pipeline():
    tokens = analyze("안녕, 세계!!! ㅋㅋㅋㅋ")
    assert tokens
    assert all(not t['pos'].startswith('S') for t in tokens)
