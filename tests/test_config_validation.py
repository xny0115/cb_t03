import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.service.service import ChatbotService


def test_start_training_invalid_config() -> None:
    svc = ChatbotService()
    svc._config["num_epochs"] = None
    from src.service import service as svc_module
    orig_read_ini = svc_module._read_ini
    try:
        svc_module._read_ini = lambda path="trainconfig.ini": {}
        res = svc.start_training("finetune")
    finally:
        svc_module._read_ini = orig_read_ini
    assert not res["success"]


def test_ini_min_lr_clamp(monkeypatch, caplog, tmp_path) -> None:
    svc = ChatbotService()
    from src.service import service as svc_module

    dummy_spm = tmp_path / "tok.model"
    dummy_spm.write_text("m")
    (tmp_path / "tok.vocab").write_text("v")
    svc._config["spm_model_path"] = str(dummy_spm)

    class DummyTok:
        def __init__(self, path: str):
            self.vocab_size = 4
            self.pad_id = 0
            self.bos_id = 1
            self.eos_id = 2
            class SP:
                def EncodeAsIds(self_inner, text: str):
                    return [0]
            self.sp = SP()
    monkeypatch.setattr(svc_module, "SentencePieceTokenizer", DummyTok)

    monkeypatch.setattr(
        svc_module, "_read_ini", lambda path="trainconfig.ini": {"train": {"min_lr": 0}}
    )
    monkeypatch.setattr(
        svc_module, "load_instruction_dataset", lambda d: []
    )
    monkeypatch.setattr(
        svc_module,
        "train_transformer",
        lambda data, cfg, save_dir=None: (object(), object()),
    )
    with caplog.at_level(logging.INFO):
        res = svc.start_training("finetune")
    assert res["success"]
    assert svc._config["min_lr"] == 1e-5
    assert "[CFG-TRAIN]" in caplog.text
