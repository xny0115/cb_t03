"""Inference helper functions."""
from __future__ import annotations

from typing import Any
import inspect
import torch


# manual decode table for fallback decoding
manual_vocab = {
    1: "<BOS>",
    3: ".",
    7: ",",
    20: "그",
    23: "안녕",
    30: "나",
    34: "잘",
    36: "다시",
    59: "끝",
    68: "은",
    79: "말",
    91: "하",
    110: "세",
    120: "지",
    122: "요",
    125: "요?",
    135: "고",
    142: "요",
    145: "니",
    191: "무",
    238: "갈",
    243: "니",
    260: "?",
    262: "다.",
    289: "게",
    316: ".",
    319: "요.",
    320: "는",
    336: "해",
    354: "뭐",
    361: "그",
    371: "해요",
    417: "요",
    428: "니?",
    219: "명절",
    25: "갖추",
    249: "바느질",
}


def decode_manual(ids: list[int]) -> str:
    """수동 매핑 테이블을 사용해 디코딩"""
    return "".join([manual_vocab.get(tok, f"[{tok}]") for tok in ids])


def _val(cfg: Any, key: str, default: Any) -> Any:
    return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)


def _encode_to_tensor(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    """커스텀·HF 토크나이저 모두 지원"""
    try:  # HuggingFace 방식
        return tokenizer.encode(text, return_tensors="pt").to(device)
    except TypeError:  # Komoran 커스텀 방식
        ids = tokenizer.encode(text)
        return torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def _safe_generate(model: Any, **kwargs: Any) -> torch.Tensor:
    """모델 generate() 시그니처에 존재하는 인자만 전달"""
    accept = inspect.signature(model.generate).parameters
    filtered = {k: v for k, v in kwargs.items() if k in accept}
    return model.generate(**filtered)


def _decode_tokens(tokenizer: Any, ids: torch.Tensor) -> str:
    """토크나이저 유형에 따라 안전하게 디코딩"""
    if isinstance(ids, torch.Tensor):
        ids = ids.squeeze(0).tolist()
    print("raw ids for decode:", ids)
    try:
        decoded = tokenizer.decode(ids)
        if not decoded.strip():
            print("\u26A0\uFE0F decode empty, fallback to manual")
            return decode_manual(ids)
        return decoded
    except Exception as e:  # pragma: no cover - 디코딩 예외 로그
        print("decode error:", e)
        return decode_manual(ids)


def generate_response(model: Any, tokenizer: Any, prompt: str, cfg: Any) -> str:
    """Return decoded response using model.generate."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None and hasattr(tokenizer, "get_vocab"):
        vocab_size = len(tokenizer.get_vocab())
    print("tokenizer vocab size:", vocab_size)
    input_ids = _encode_to_tensor(tokenizer, prompt, device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    amp = bool(_val(cfg, "use_mixed_precision", False)) and device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=amp):
        gen_kwargs = dict(
            src=input_ids,
            max_length=_val(cfg, "max_length", 128),
            temperature=_val(cfg, "temperature", 1.0),
            top_k=_val(cfg, "top_k", 0),
            top_p=_val(cfg, "top_p", 1.0),
            no_repeat_ngram_size=_val(cfg, "no_repeat_ngram_size", 0),
            num_beams=_val(cfg, "num_beams", 1),
            repetition_penalty=_val(cfg, "repetition_penalty", 1.1),
        )
        model.eval()
        output = _safe_generate(model, **gen_kwargs)
        print("raw output tensor:", output.tolist())
    decoded = _decode_tokens(tokenizer, output[0])
    print("\u2705 decoded result:", decoded)
    return decoded
