import json
import sys
import types
import importlib


def _setup_stub(monkeypatch):
    module = types.ModuleType("src.data.morph")
    module.analyze = lambda text: [{"text": text, "lemma": text, "pos": "NNG"}]
    monkeypatch.setitem(sys.modules, "src.data.morph", module)
    importlib.reload(importlib.import_module("purifier.purifier"))


def test_clean_file_strips_quotes(tmp_path, monkeypatch):
    _setup_stub(monkeypatch)
    from purifier.purifier import clean_file

    raw = tmp_path / "raw.json"
    raw.write_text('[{"question": "\\"명절\\"은 언제야?", "answer": "\\"추석\\"이야"}]', encoding="utf-8")

    out = clean_file(raw, tmp_path)
    data = json.loads(out.read_text(encoding="utf-8"))
    q_text = data[0]["question"]["text"]
    a_text = data[0]["answer"]["text"]
    assert '"' not in q_text
    assert '"' not in a_text
    for tok in data[0]["question"]["tokens"] + data[0]["answer"]["tokens"]:
        assert '"' not in tok["text"]
        assert '"' not in tok["lemma"]
