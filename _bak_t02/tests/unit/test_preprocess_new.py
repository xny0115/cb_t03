import pytest
pytest.importorskip("soynlp")
from pathlib import Path
from src.data import QADataset
from src.data.preprocess import extract_concepts, infer_domain
from src.data.morph import analyze

pytestmark = pytest.mark.requires_torch


def test_concept_extraction():
    tokens = analyze("삼겹살이 맛있어")
    concepts = extract_concepts(tokens)
    assert "삼겹살" in concepts


def test_domain_inference():
    tokens = analyze("삼겹살이 맛있어")
    domain = infer_domain(tokens)
    assert domain == "고기"


def test_dataset_returns_text():
    ds = QADataset(Path("datas"))
    q, a = ds[0]
    assert q == ds.pairs[0].question
    assert a == ds.pairs[0].answer
