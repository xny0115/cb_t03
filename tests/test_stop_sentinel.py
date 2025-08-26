# CPU 전용 테스트(gpu 없음 환경용)
# STOP 센티넬 생성 시 학습 루프 중단 및 백엔드 응답을 검증한다.

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training import simple
from src.ui.backend import WebBackend
from src.service.service import ChatbotService


def test_stop_file_breaks_epoch(tmp_path, monkeypatch):
    calls = {"n": 0}

    class DummyTok:
        def __init__(self, path: str):
            self.vocab_size = 8
            self.pad_id = 0
            self.bos_id = 1
            self.eos_id = 2
            self._last = ""
            class SP:
                def GetPieceSize(self_inner):
                    return 4
                def EncodeAsIds(self_inner, text: str):
                    return [1, 2]
                def IdToPiece(self_inner, idx: int):
                    return "<unk>"
            self.sp = SP()
        def encode(self, text: str, add_special: bool = False):
            self._last = text
            return [1, 2, 3]
        def decode(self, ids):
            return self._last
    monkeypatch.setattr(simple, "SentencePieceTokenizer", DummyTok)

    class DummyModel:
        def __init__(self, vocab: int):
            class Embed:
                def __init__(self, n):
                    self.num_embeddings = n
            class Fc:
                def __init__(self, n):
                    self.out_features = n
            self.embed = Embed(vocab)
            self.fc_out = Fc(vocab)
        def state_dict(self):
            return {}
    class DummyScaler:
        def step(self, opt):
            pass
        def update(self):
            pass
        def state_dict(self):
            return {}
    def fake_init_model(tok, cfg):
        import torch
        opt = torch.optim.SGD([torch.tensor(0.0, requires_grad=True)], lr=0.1)
        return DummyModel(tok.vocab_size), object(), opt, DummyScaler(), "cpu", False
    monkeypatch.setattr(simple, "_init_model", fake_init_model)

    def fake_prepare(samples, tokenizer, is_pretrain):
        class DS:
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return ([0], [0])
        return DS(), 1
    monkeypatch.setattr(simple, "_prepare_dataset", fake_prepare)
    monkeypatch.setattr(simple, "_create_loader", lambda dataset, cfg, drop_last=True: [([0], [0])])

    def fake_train_epoch(*args, **kwargs):
        calls["n"] += 1
        return 0.0, 0.0, None, None
    monkeypatch.setattr(simple, "_train_epoch", fake_train_epoch)

    def fake_save_checkpoint(state, path):
        if calls["n"] == 1:
            (tmp_path / "STOP").touch()
    monkeypatch.setattr(simple, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(simple, "save_transformer", lambda model, meta, path: None)

    cfg = {"num_epochs": 2, "spm_model_path": str(tmp_path / "spm.model")}
    (tmp_path / "spm.model").write_text("m")
    (tmp_path / "spm.vocab").write_text("v")
    sample = simple.InstructionSample("", "", "x")
    simple.train([sample], cfg, is_pretrain=False, save_dir=tmp_path)
    assert calls["n"] == 1


def test_backend_delete_model(monkeypatch):
    svc = ChatbotService()
    backend = WebBackend(svc)
    monkeypatch.setattr(svc, "delete_model", lambda: True)
    res = backend.delete_model()
    assert res["msg"] == "stop_requested" and res["success"]
