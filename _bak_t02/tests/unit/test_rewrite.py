import pytest
pytest.importorskip("soynlp")
from src.service.rewriter import rewrite

pytestmark = pytest.mark.requires_torch


def test_rewrite_basic():
    out = rewrite("돼지고기에 대해 알려줘.")
    assert isinstance(out, str) and out

