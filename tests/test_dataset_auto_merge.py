import json
from pathlib import Path

from src.data.loader import load_instruction_dataset, load_pretrain_dataset
from src.service.service import ChatbotService


def test_dataset_auto_merge(tmp_path: Path) -> None:
    pre = tmp_path / "pretrain"
    ft = tmp_path / "finetune"
    add = tmp_path / "additional_finetune"
    pre.mkdir()
    ft.mkdir()
    add.mkdir()

    (pre / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    (pre / "b.txt").write_text("world\nnew\n", encoding="utf-8")

    sample = {"instruction": "인사", "input": "", "output": "안녕"}
    (ft / "a.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    (ft / "b.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    assert load_pretrain_dataset(pre) == ["hello", "world", "new"]
    assert len(load_instruction_dataset(ft)) == 1

    svc = ChatbotService()
    svc.pretrain_dir = pre
    svc.finetune_dir = ft
    svc.additional_dir = add
    svc.model_dir = tmp_path / "models"
    svc._config.update({"num_epochs": 1, "model_dim": 32, "num_encoder_layers": 1, "num_decoder_layers": 1})
    svc.start_training("finetune")

    assert svc.model is not None and len(svc.dataset) == 1
