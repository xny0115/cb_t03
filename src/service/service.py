# === LLM SERVICE DIRECTION ===============================
# 이 엔드포인트/모델 서빙은 LLM 기준으로 설계됩니다.
# 추론 로직은 토큰화/컨텍스트 길이/마스크 일관성을 우선 검증합니다.
# =========================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List
import logging
import os
import hashlib
import re
import configparser
import torch

try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
except ImportError:
    pynvml = None


from ..data.loader import (
    load_instruction_dataset,
    load_pretrain_dataset,
    get_dataset_info,
)
from ..config import load_config, save_config
from ..model import (
    DummyModel,
    HFModel,
    load_model,
    Seq2SeqTransformer,
    save_transformer,
    load_transformer,
)
from ..training.simple import train as train_transformer, pretrain as train_pretrain
from ..utils.tokenizer import SentencePieceTokenizer
from ..utils.validator import validate_config
from ..tuning.auto import AutoTuner


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_epoch_metrics(fp: Path) -> List[str]:
    if not fp.exists():
        return []
    lines = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        if "Epoch" in line and "Loss" in line:
            lines.append(line.strip())
    return lines


def _param_count(model) -> int:
    try:
        params = getattr(model, 'parameters', None)
        return sum(p.numel() for p in params()) if callable(params) else 0
    except Exception:
        return 0


def _read_ini(path: str = "trainconfig.ini") -> dict:
    """trainconfig.ini에서 학습/추론 설정을 읽어 dict로 반환."""
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not cfg.read(path, encoding="utf-8"):
        return {}

    def get(section: str, key: str, cast, default=None):
        v = cfg.get(section, key, fallback=None)
        if v is None or str(v).strip() == "":
            return default
        if cast is bool:
            return str(v).strip().lower() in ("1", "true", "yes", "y", "on")
        if cast is int:
            return int(float(v))
        if cast is float:
            return float(v)
        return v

    def parse_train(section: str) -> Dict[str, Any]:
        return {
            k: v
            for k, v in {
                "epochs": get(section, "epochs", int, None),
                "num_epochs": get(section, "num_epochs", int, None),
                "batch_size": get(section, "batch_size", int, None),
                "learning_rate": get(section, "learning_rate", float, None),
                "dropout_ratio": get(section, "dropout_ratio", float, None),
                "grad_clip": get(section, "grad_clip", float, None),
                "min_lr": get(section, "min_lr", float, None),
                "use_mixed_precision": get(section, "use_mixed_precision", bool, None),
                "model_dim": get(section, "model_dim", int, None),
                "ff_dim": get(section, "ff_dim", int, None),
                "num_heads": get(section, "num_heads", int, None),
                "num_encoder_layers": get(section, "num_encoder_layers", int, None),
                "num_decoder_layers": get(section, "num_decoder_layers", int, None),
                "num_workers": get(section, "num_workers", int, None),
                "pin_memory": get(section, "pin_memory", bool, None),
                "spm_model_path": get(section, "spm_model_path", str, None),
                "resume": get(section, "resume", bool, False),
            }.items()
            if v is not None
        }

    gen = {
        "lock_ui": get("generate", "lock_ui", bool, True),
        "temperature": get("generate", "temperature", float, None),
        "top_p": get("generate", "top_p", float, None),
        "max_new_tokens": get("generate", "max_new_tokens", int, None),
        "repetition_penalty": get("generate", "repetition_penalty", float, None),
        "top_k": get("generate", "top_k", int, None),
        "no_repeat_ngram_size": get("generate", "no_repeat_ngram_size", int, None),
        "num_beams": get("generate", "num_beams", int, None),
        "do_sample": get("generate", "do_sample", bool, None),
        "seed": get("generate", "seed", str, None),
        "stop": get("generate", "stop", str, None),
    }

    out = {"generate": {k: v for k, v in gen.items() if v is not None}, "train": parse_train("train")}
    for sec in ("pretrain", "finetune"):
        if cfg.has_section(sec):
            out[sec] = parse_train(sec)
    return out


def _resolve_generate(_cfg: dict | None = None) -> dict:
    """INI 설정을 기준으로 추론 파라미터 확정."""
    ini = _read_ini().get("generate", {})

    def pick(name: str, default, lo=None, hi=None, cast=float):
        v = ini.get(name, default)
        try:
            v = cast(v) if v is not None else default
        except Exception:
            v = default
        if isinstance(v, (int, float)) and lo is not None and hi is not None:
            v = max(lo, min(hi, v))
        return v

    p = {
        "temperature": pick("temperature", 0.3, 0.0, 2.0, float),
        "top_p": pick("top_p", 0.9, 0.0, 1.0, float),
        "max_new_tokens": pick("max_new_tokens", 128, 1, 4096, int),
        "repetition_penalty": pick("repetition_penalty", 1.10, 0.8, 2.0, float),
        "top_k": pick("top_k", 0, 0, 100, int),
        "no_repeat_ngram_size": pick("no_repeat_ngram_size", 0, 0, 10, int),
        "num_beams": pick("num_beams", 1, 1, 8, int),
        "do_sample": bool(pick("do_sample", True, None, None, bool)),
        "seed": ini.get("seed", "auto"),
        "stop": ini.get("stop", None),
    }
    if p["num_beams"] > 1:
        p["do_sample"] = False
    return p


def _apply_train_ini(cfg: dict) -> dict:
    """train 섹션 값을 기존 config에 덮어쓰기."""
    over = _read_ini().get("train", {})
    out = dict(cfg)
    if "epochs" in over:
        out["num_epochs"] = over.pop("epochs")
    out.update(over)
    return out

class ChatbotService:
    """Instruction 기반 챗봇 서비스."""

    MAX_INPUT_LEN = 1000

    def __init__(self) -> None:
        self.data_dir = Path("datas")
        self.pretrain_dir = self.data_dir / "pretrain"
        self.finetune_dir = self.data_dir / "finetune"
        self.model_dir = Path("models")
        self.model_path = self.model_dir / "finetune.pth"
        self.dataset = load_instruction_dataset(self.finetune_dir)
        self.model: DummyModel | HFModel | Seq2SeqTransformer | None = None
        self.tokenizer: SentencePieceTokenizer | None = None
        self._config = load_config()

        if pynvml:
            try:
                pynvml.nvmlInit()
                self._pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._pynvml_handle = None
        else:
            self._pynvml_handle = None


        hf_name = os.getenv("HF_MODEL_NAME")
        if hf_name:
            self.model = HFModel(hf_name)
        elif self.model_path.exists():
            try:
                self.model, _ = load_transformer(self.model_path)
            except Exception:
                self.model = load_model(self.model_path)
            logging.getLogger(__name__).info(
                "[SERVE] model_loaded path=%s params=%s",
                str(self.model_path),
                _param_count(self.model),
            )
            spm_model_path = str(self._config.get("spm_model_path", "models/spm.model"))
            spm_path = Path(spm_model_path)
            if not spm_path.is_absolute():
                spm_path = (Path.cwd() / spm_path).resolve()
            if spm_path.exists():
                self.tokenizer = SentencePieceTokenizer(str(spm_path))
                ckpt = self.model_dir / f"last_{self.model_path.stem}.ckpt"
                if ckpt.exists():
                    info = torch.load(ckpt, map_location="cpu").get("tokenizer_info", {})
                    ps = info.get("piece_size")
                    ckpt_sha = info.get("sha256")
                    model_b = Path(spm_path).read_bytes()
                    vocab_b = Path(spm_path.with_suffix(".vocab")).read_bytes()
                    # SPM 모델과 어휘를 결합해 해시 계산
                    sha = hashlib.sha256(model_b + vocab_b).hexdigest()
                    if ps is not None and ckpt_sha and (
                        ps != self.tokenizer.sp.GetPieceSize() or ckpt_sha != sha
                    ):
                        msg = "spm mismatch: piece_size/sha256 differ"
                        logging.getLogger(__name__).error("[SERVE] %s", msg)
                        raise RuntimeError(msg)
            else:
                logging.getLogger(__name__).warning(f"SPM model not found at {spm_path}, tokenizer not loaded.")

    def start_training(self, mode: str) -> Dict[str, Any]:
        """학습 유형에 따라 분기 처리."""
        self._config = _apply_train_ini(self._config)
        ini = _read_ini()
        if mode == "pretrain" and "pretrain" in ini:
            over = dict(ini["pretrain"])
            if "epochs" in over:
                over["num_epochs"] = over.pop("epochs")
            self._config.update(over)
        elif mode == "finetune" and "finetune" in ini:
            over = dict(ini["finetune"])
            if "epochs" in over:
                over["num_epochs"] = over.pop("epochs")
            self._config.update(over)
        valid, msg = validate_config(self._config)
        if not valid:
            return {"success": False, "msg": msg, "data": None}
        overrides = ini.get("train", {})
        checks = [
            ("num_epochs", lambda v: v >= 1),
            ("batch_size", lambda v: v >= 1),
            ("learning_rate", lambda v: 1e-6 <= v <= 1e-2),
            ("dropout_ratio", lambda v: 0 <= v < 0.9),
            ("grad_clip", lambda v: 0 <= v <= 10),
            ("min_lr", lambda v: 1e-6 <= v <= 1e-3),
            ("model_dim", lambda v: v > 0),
            ("ff_dim", lambda v: v > 0),
            ("num_heads", lambda v: v > 0),
            ("num_encoder_layers", lambda v: v > 0),
            ("num_decoder_layers", lambda v: v > 0),
        ]
        for key, rule in checks:
            if key in overrides:
                val = self._config.get(key)
                if val is None or not rule(val):
                    return {"success": False, "msg": f"{key}: out_of_range", "data": None}
        if isinstance(self.model, HFModel):
            return {"success": True, "msg": "done", "data": None}

        logger = logging.getLogger(__name__)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("CUDA not available: proceeding on CPU (slow).")

        logger.info("training mode=%s, device=%s", mode, device)

        cfg = self._config
        cfg["resume"] = bool(cfg.get("resume", False) and mode == "resume")
        logger.info(
            "[CFG-TRAIN] epochs=%s batch=%s lr=%s dropout=%s amp=%s grad_clip=%s resume=%s",
            cfg.get('num_epochs'),
            cfg.get('batch_size'),
            cfg.get('learning_rate'),
            cfg.get('dropout_ratio'),
            cfg.get('use_mixed_precision'),
            cfg.get('grad_clip'),
            cfg.get('resume'),
        )

        resume = bool(cfg.get("resume", False))
        try:
            if mode == "pretrain":
                from ..data.subtitle_cleaner import clean_subtitle_files
                clean_subtitle_files(Path("."), self.pretrain_dir)
                data = load_pretrain_dataset(self.pretrain_dir)
                logger.info(
                    "[DATA] pretrain size=%s",
                    (len(data) if hasattr(data, "__len__") else "?"),
                )
                self.model_path = self.model_dir / "pretrain.pth"
                model, tokenizer = train_pretrain(
                    data, cfg, save_dir=str(self.model_path.parent)
                )
            elif resume:
                is_pretrain_resume = (self.model_dir / "pretrain.pth").exists() or (
                    self.model_dir / "last_pretrain.ckpt"
                ).exists()

                cfg["resume"] = True

                if is_pretrain_resume:
                    data = load_pretrain_dataset(self.pretrain_dir)
                    self.model_path = self.model_dir / "pretrain.pth"
                    cfg["model_path"] = str(self.model_path)
                    model, tokenizer = train_pretrain(
                        data, cfg, save_dir=str(self.model_path.parent)
                    )
                else:
                    data = load_instruction_dataset(self.finetune_dir)
                    self.model_path = self.model_dir / "finetune.pth"
                    cfg["model_path"] = str(self.model_path)
                    model, tokenizer = train_transformer(
                        data, cfg, save_dir=str(self.model_path.parent)
                    )
            else:
                data = load_instruction_dataset(self.finetune_dir)
                logger.info(
                    "[DATA] finetune size=%s",
                    (len(data) if hasattr(data, "__len__") else "?"),
                )
                self.model_path = self.model_dir / "finetune.pth"
                model, tokenizer = train_transformer(
                    data, cfg, save_dir=str(self.model_path.parent)
                )
        except Exception as exc:
            logger.warning("training failed: %s", exc)
            return {"success": False, "msg": str(exc), "data": None}

        # Note: save_transformer is now called inside train()

        self.model = model
        self.tokenizer = tokenizer
        logger.info("Training complete")
        return {"success": True, "msg": "done", "data": None}

    def set_config(self, cfg: Dict[str, Any]) -> Tuple[bool, str]:
        valid, msg = validate_config(cfg)
        if not valid:
            return False, msg
        try:
            self._config.update(cfg)
            save_config(self._config)
            return True, "saved"
        except Exception as exc:
            logging.getLogger(__name__).warning("config save failed: %s", exc)
            return False, str(exc)

    def get_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def delete_model(self) -> bool:
        deleted = False
        for fp in self.model_dir.glob("*.pth"):
            try:
                fp.unlink()
                deleted = True
            except FileNotFoundError:
                continue
            except Exception as exc:
                logging.getLogger(__name__).warning("model delete failed: %s", exc)
        if deleted:
            self.model = None
            self.tokenizer = None
        return deleted

    def infer(self, text: str) -> Dict[str, Any]:
        if not self.model:
            return {"success": False, "msg": "no_model", "data": None}
        if isinstance(self.model, Seq2SeqTransformer) and not self.tokenizer:
            return {"success": False, "msg": "no_tokenizer", "data": None}
        if not text.strip():
            return {"success": False, "msg": "empty_input", "data": None}
        if len(text) > self.MAX_INPUT_LEN:
            return {"success": False, "msg": "too_long", "data": None}
        params = _resolve_generate()
        logging.getLogger(__name__).info(
            "[CFG-GEN] t=%.2f tp=%.2f k=%d mnt=%d rep=%.2f beams=%d sample=%s",
            params["temperature"],
            params["top_p"],
            params["top_k"],
            params["max_new_tokens"],
            params["repetition_penalty"],
            params["num_beams"],
            str(params["do_sample"]),
        )

        if isinstance(self.model, HFModel):
            generator = None
            if params["seed"] != "auto":
                try:
                    device_str = "cuda" if torch.cuda.is_available() else "cpu"
                    generator = torch.Generator(device=device_str).manual_seed(
                        int(params["seed"])
                    )
                except Exception:
                    generator = None
            kwargs = {
                "temperature": params["temperature"],
                "top_p": params["top_p"],
                "top_k": params["top_k"],
                "max_new_tokens": params["max_new_tokens"],
                "repetition_penalty": params["repetition_penalty"],
                "no_repeat_ngram_size": params["no_repeat_ngram_size"],
                "num_beams": params["num_beams"],
                "do_sample": params["do_sample"],
            }
            if generator:
                kwargs["generator"] = generator
            if params.get("stop"):
                kwargs["stop_sequence"] = params["stop"]
            outputs = self.model.pipe(text, **kwargs)
            if isinstance(outputs, list) and outputs:
                out_text = outputs[0].get("generated_text", "").strip()
            else:
                out_text = str(outputs).strip()
            msg = "ok" if out_text else "no_answer"
            return {"success": True, "msg": msg, "data": out_text}

        if isinstance(self.model, Seq2SeqTransformer) and self.tokenizer:
            ids = self.tokenizer.encode(text, True)
            src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            src = src.to(next(self.model.parameters()).device)
            out_ids = self.model.generate(
                src,
                max_new_tokens=params["max_new_tokens"],
                eos_id=self.tokenizer.eos_id,
                pad_id=self.tokenizer.pad_id,
            )
            out_text = self.tokenizer.decode(out_ids.squeeze().tolist())
            msg = "ok" if out_text else "no_answer"
            return {"success": True, "msg": msg, "data": out_text}

        out = self.model.predict("", text)
        if not out:
            return {"success": True, "msg": "no_answer", "data": ""}
        return {"success": True, "msg": "ok", "data": out}

    def get_status(self) -> Dict[str, Any]:
        data = {}
        if psutil:
            data["cpu_usage"] = psutil.cpu_percent()
        else:
            data["cpu_usage"] = 0.0

        if self._pynvml_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._pynvml_handle)
                data["gpu_usage"] = float(util.gpu)
            except Exception:
                data["gpu_usage"] = None
        else:
            data["gpu_usage"] = None

        return {"success": True, "msg": "idle", "data": data}

    def auto_tune(self) -> Dict[str, Any]:
        """Apply AutoTuner suggestions to config."""
        size, tokens, txt_lines, json_lines, skipped = get_dataset_info(
            self.pretrain_dir, self.finetune_dir, self.data_dir / "additional"
        )
        print(f"[DEBUG] Found pretrain txt files: {txt_lines} lines")
        print(f"[DEBUG] Found finetune jsonl files: {json_lines} lines")
        if skipped:
            print(f"[DEBUG] Skipped files: {skipped}")
        print(f"[DEBUG] AutoTune triggered: dataset size = {size}, tokens = {tokens}")
        cfg = AutoTuner(size, tokens).suggest()
        valid, msg = validate_config(cfg)
        if not valid:
            return {"success": False, "msg": msg, "data": None}
        self._config.update(cfg)
        save_config(self._config)
        logging.getLogger(__name__).info("auto-tune applied: %s", cfg)
        return {"success": True, "msg": "auto-tune applied", "data": cfg}
