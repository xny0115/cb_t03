from __future__ import annotations
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, Tuple
from collections import defaultdict, Counter
import json
import logging
import os

try:
    import psutil  # type: ignore
except Exception:
    psutil = None
try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None
from ..config import CONFIG_PATH, load_config, apply_schema
from .utils import to_config, normalize_config, _CFG_MAP
from ..utils.logger import LOG_PATH
from ..utils.vocab import Tokenizer
from ..data.loader import load_all
from ..tuning.auto import suggest_config
from .loader import load_model
from .rewriter import rewrite as rewrite_sentence
from .train_ops import run_training, start_training
from .infer_ops import infer_text, rewrite_text, get_status, delete_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChatbotService:
    def __init__(self) -> None:
        self._config = load_config()
        self.training = False
        self.model_path = Path(os.environ.get("MODEL_PATH", "models/current.pth"))
        self.model_exists = self.model_path.exists()
        self._status_msg = "idle"
        self._progress = 0.0
        self._last_loss = 0.0
        self._tokenizer: Tokenizer | None = None
        self._model = None
        self._lock = Lock()
        self._thread: Thread | None = None
        data_dir = Path(os.environ.get("DATA_DIR", "datas"))
        self._dataset = load_all(data_dir)
        removed = 0
        valid = []
        for item in self._dataset:
            q = item.question.get("text")
            a = item.answer.get("text")
            tq = item.question.get("tokens", [])
            ta = item.answer.get("tokens", [])
            if not q or not a or not tq or not ta or len(item.concepts) == 0:
                removed += 1
                continue
            valid.append(item)
        if removed:
            logger.info("invalid dataset entries removed: %d", removed)
        self._dataset = valid
        pos_cnt: dict[str, Counter] = defaultdict(Counter)
        for item in self._dataset:
            for tok in item.question.get("tokens", []) + item.answer.get("tokens", []):
                lem = tok.get("lemma")
                pos = tok.get("pos")
                if lem and pos:
                    pos_cnt[lem][pos] += 1
        self._pos_map = {k: c.most_common(1)[0][0] for k, c in pos_cnt.items()}
        self.auto_tune()
        if self.model_exists:
            self._tokenizer, self._model = load_model(self.model_path)
            self._model.eval()
        else:
            logger.warning("model file missing: %s", self.model_path)
        logger.info("Service initialized")

    def auto_tune(self) -> dict:  # pragma: no cover
        num_pairs = len(self._dataset)
        try:
            cfg = suggest_config(num_pairs)
            self._config.update(cfg)
            logger.info("auto-tune applied: %s", cfg)
            return cfg
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("auto-tune failed: %s", exc)
            return {}

    def get_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def update_config(self, cfg: Dict[str, Any]) -> Tuple[bool, str]:
        """Update config dict and persist with type casting."""
        try:
            with self._lock:
                # map dataclass-style keys to internal names
                for key, (src, _) in _CFG_MAP.items():
                    if key in cfg:
                        cfg[src] = cfg[key]
                self._config.update(cfg)
                self._config = apply_schema(self._config)
                # force type for critical params
                self._config["epochs"] = int(cfg.get("epochs", 20))
                self._config["batch_size"] = int(cfg.get("batch_size", 4))
                self._config["lr"] = float(cfg.get("lr", 1e-4))
                norm = normalize_config(to_config(self._config)).__dict__
                for key, (src, _) in _CFG_MAP.items():
                    self._config[src] = norm[key]
                json.dump(self._config, open(CONFIG_PATH, "w"), indent=2)
            return True, "saved"
        except Exception as exc:
            logger.exception("config save failed")
            return False, str(exc)

    def train(self, path: Path) -> None:  # pragma: no cover
        """Run synchronous training via helper."""
        run_training(self, path)

    def start_training(self, data_path: str = ".") -> Dict[str, Any]:
        """Start background training via helper."""
        return start_training(self, data_path)

    def stop_training(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.info("Stop requested, waiting for thread")
            self._thread.join()

    def infer(self, text: str) -> Dict[str, Any]:
        return infer_text(self, text)

    def rewrite(self, text: str) -> Dict[str, Any]:
        """Return paraphrased sentence referencing dataset."""
        return rewrite_text(self, text)

    def get_status(self) -> Dict[str, Any]:
        return get_status(self)

    def delete_model(self) -> bool:  # pragma: no cover
        return delete_model(self)
