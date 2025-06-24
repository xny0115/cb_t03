from pathlib import Path

import torch

from src.training.simple import train
from src.model.transformer import save_transformer, load_transformer
from src.data.loader import InstructionSample


def test_transformer_save_load(tmp_path: Path) -> None:
    samples = [InstructionSample("다음 질문에 답하세요.", "안녕", "안녕!")]
    model, tokenizer = train(samples, epochs=1)
    path = tmp_path / "m.pth"
    save_transformer(model, tokenizer.stoi, path)
    assert path.exists() and path.stat().st_size >= 1_000_000
    loaded, vocab = load_transformer(path)
    assert isinstance(vocab, dict) and len(vocab) == len(tokenizer.stoi)
    ids = tokenizer.encode("안녕", True)
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    out_ids = loaded.generate(src, max_new_tokens=5, eos_id=tokenizer.stoi["<eos>"])
    assert out_ids.size(0) == 1

