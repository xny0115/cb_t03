from pathlib import Path

import torch

from src.training.simple import train
import pytest
from src.model.transformer import save_transformer, load_transformer
from src.data.loader import InstructionSample
from src.utils.tokenizer import SentencePieceTokenizer


@pytest.mark.gpu
def test_transformer_save_load(tmp_path: Path) -> None:
    samples = [InstructionSample("다음 질문에 답하세요.", "안녕", "안녕!")]
    tok = SentencePieceTokenizer(Path("models/spm_bpe_8k.model"))
    model = train(samples, {"num_epochs": 1, "tokenizer_path": str(tok.model_path)})
    path = tmp_path / "m.pth"
    save_transformer(model, path)
    assert path.exists() and path.stat().st_size >= 1_000_000
    loaded = load_transformer(path)
    ids = tok.encode("안녕", True)
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    out_ids = loaded.generate(src, max_new_tokens=5, eos_id=tok.eos_id)
    assert out_ids.size(0) == 1

