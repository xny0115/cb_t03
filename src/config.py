"""프로젝트 설정 도우미."""
from __future__ import annotations

import configparser
import os
import re
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "trainconfig.ini"))

DEFAULT_CONFIG: Dict[str, Any] = {
    "num_epochs": 20,
    "batch_size": 48,
    "learning_rate": 2e-4,
    "dropout_ratio": 0.1,
    "grad_clip": 1.0,
    "min_lr": 1e-5,
    "warmup_steps": 0,
    "max_sequence_length": 128,
    "num_heads": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "model_dim": 256,
    "ff_dim": 1024,
    "top_k": 10,
    "temperature": 0.7,
    "top_p": 0.9,
    "data_preprocessing": "none",
    "embedding_dim": 256,
    "activation_function": "relu",
    "optimizer_selection": "adam",
    "lr_scheduler": "none",
    "normalization_technique": "layer_norm",
    "attention_type": "multi_head",
    "positional_encoding": "sine",
    "gradient_clipping": 1.0,
    "weight_decay": 0.01,
    "early_stopping": True,
    "early_stopping_patience": 8,
    "save_every": 0,
    "num_workers": 6,
    "pin_memory": True,
    "use_mixed_precision": True,
    "repetition_penalty": 1.1,
    "max_response_length": 64,
    "beam_search_size": 3,
    "diversity_penalty": 0.5,
    "pattern_recognition": False,
    "spm_model_path": "models/spm.model",
    "resume": False,
}

_BOOL_TRUE = {"1", "true", "yes", "y", "on"}
_TRAIN_SECTION_KEYS = (
    "grad_clip",
    "min_lr",
    "use_mixed_precision",
    "model_dim",
    "ff_dim",
    "num_heads",
    "num_encoder_layers",
    "num_decoder_layers",
    "num_workers",
    "pin_memory",
    "spm_model_path",
)
_MODE_KEY_MAP = {
    "num_epochs": "epochs",
    "batch_size": "batch_size",
    "learning_rate": "learning_rate",
    "dropout_ratio": "dropout_ratio",
    "resume": "resume",
}


def _cast_value(raw: str, default: Any) -> Any:
    """INI 문자열을 기본값 타입에 맞춰 변환."""
    if isinstance(default, bool):
        return raw.strip().lower() in _BOOL_TRUE
    if isinstance(default, int):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default
    return raw.strip()


def _format_value(value: Any) -> str:
    """INI에 기록할 문자열 표현을 반환."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        text = f"{value:.10f}".rstrip('0').rstrip('.')
        return text or '0'
    return str(value)


def _ensure_section(text: str, section: str) -> str:
    """섹션이 없으면 새로 추가."""
    pattern = re.compile(rf"^\[{re.escape(section)}\]", re.MULTILINE)
    if pattern.search(text):
        return text
    suffix = "\n" if text and not text.endswith("\n") else ""
    return f"{text}{suffix}[{section}]\n"


def _replace_in_section(text: str, section: str, key: str, value: str) -> str:
    """특정 섹션에서 키 값을 교체하거나 새로 추가."""
    section_pattern = re.compile(
        rf"(^\[{re.escape(section)}\][^\n]*\n)(?P<body>.*?)(?=^\[|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = section_pattern.search(text)
    if not match:
        text = _ensure_section(text, section)
        match = section_pattern.search(text)
        if not match:
            # 섹션 생성 후에도 발견되지 않으면 끝에 값 추가
            suffix = "\n" if text and not text.endswith("\n") else ""
            return f"{text}{suffix}{key} = {value}\n"

    body = match.group("body")
    key_pattern = re.compile(
        rf"(?m)^(?P<indent>\s*{re.escape(key)}\s*=\s*)(?P<val>[^\n#;]*?)(?P<trail>\s*)(?P<comment>[#;].*)?$"
    )

    def repl(m: re.Match[str]) -> str:
        indent = m.group("indent")
        trail = m.group("trail")
        comment = m.group("comment") or ""
        return f"{indent}{value}{trail}{comment}"

    new_body, count = key_pattern.subn(repl, body, count=1)
    if count == 0:
        insertion = f"{key} = {value}\n"
        if not body:
            new_body = insertion
        else:
            base = body if body.endswith("\n") else f"{body}\n"
            new_body = f"{base}{insertion}"

    start, end = match.start("body"), match.end("body")
    return f"{text[:start]}{new_body}{text[end:]}"


def load_config() -> Dict[str, Any]:
    """trainconfig.ini를 읽어 기본 설정과 병합."""
    cfg = DEFAULT_CONFIG.copy()
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    try:
        read_files = parser.read(CONFIG_PATH, encoding="utf-8")
    except (OSError, configparser.Error):
        read_files = []
    if not read_files:
        return cfg

    if parser.has_section("train"):
        for key in _TRAIN_SECTION_KEYS:
            if parser.has_option("train", key):
                raw = parser.get("train", key, fallback="")
                cfg[key] = _cast_value(raw, DEFAULT_CONFIG.get(key))

    loaded_from_pretrain: set[str] = set()
    if parser.has_section("pretrain"):
        for key, ini_key in _MODE_KEY_MAP.items():
            if parser.has_option("pretrain", ini_key):
                raw = parser.get("pretrain", ini_key, fallback="")
                cfg[key] = _cast_value(raw, DEFAULT_CONFIG.get(key))
                loaded_from_pretrain.add(key)

    if parser.has_section("finetune"):
        for key, ini_key in _MODE_KEY_MAP.items():
            if key in loaded_from_pretrain:
                continue
            if parser.has_option("finetune", ini_key):
                raw = parser.get("finetune", ini_key, fallback="")
                cfg[key] = _cast_value(raw, DEFAULT_CONFIG.get(key))

    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    """trainconfig.ini 파일에 설정을 반영."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"trainconfig.ini not found: {CONFIG_PATH}")

    text = CONFIG_PATH.read_text(encoding="utf-8")
    for key in _TRAIN_SECTION_KEYS:
        if key in cfg and cfg[key] is not None:
            value = _format_value(cfg[key])
            text = _replace_in_section(text, "train", key, value)

    for key, ini_key in _MODE_KEY_MAP.items():
        if key in cfg and cfg[key] is not None:
            value = _format_value(cfg[key])
            text = _replace_in_section(text, "pretrain", ini_key, value)
            text = _replace_in_section(text, "finetune", ini_key, value)

    if not text.endswith("\n"):
        text = f"{text}\n"
    CONFIG_PATH.write_text(text, encoding="utf-8")
