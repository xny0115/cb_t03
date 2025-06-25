"""Auto tuning utilities."""
from __future__ import annotations

import logging
from typing import Any, Dict

from ..utils.validator import REQUIRED_KEYS

import torch
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None

logger = logging.getLogger(__name__)


class AutoTuner:
    """Suggest config based on dataset size and hardware."""

    def __init__(self, dataset_size: int, token_count: int | None = None) -> None:
        self.dataset_size = dataset_size
        self.token_count = token_count
        self.vram_gb = self._get_vram()
        self.ram_gb = self._get_ram()

    def _get_vram(self) -> int:
        try:
            if torch.cuda.is_available():
                prop = torch.cuda.get_device_properties(0)
                return prop.total_memory // 1_073_741_824
        except Exception:  # pragma: no cover - best effort
            logger.debug("vram detection failed")
        return 0

    def _get_ram(self) -> int:
        if psutil:
            try:
                return psutil.virtual_memory().total // 1_073_741_824
            except Exception:  # pragma: no cover - best effort
                logger.debug("ram detection failed")
        return 0

    def suggest(self) -> Dict[str, Any]:
        """Return recommended hyperparameters with all required keys."""
        cfg: Dict[str, Any] = {}
        tok_msg = (
            f", tokens={self.token_count}" if self.token_count is not None else ""
        )
        logger.info(
            "Auto-Tune 기준: dataset size=%d samples%s, vram=%dGB, ram=%dGB",
            self.dataset_size,
            tok_msg,
            self.vram_gb,
            self.ram_gb,
        )
        high_mem = self.vram_gb >= 8 or self.ram_gb >= 16
        model_dim = 512 if self.vram_gb >= 8 else 256
        num_layers = 6 if self.dataset_size > 500 else 4
        cfg.update(
            {
                "batch_size": 32 if high_mem else 8,
                "model_dim": model_dim,
                "ff_dim": 2048 if model_dim >= 512 else 1024,
                "num_encoder_layers": num_layers,
                "num_decoder_layers": num_layers,
                "num_epochs": 30 if self.dataset_size > 300 else 15,
                "learning_rate": 5e-4 if self.dataset_size > 300 else 1e-3,
                "dropout_ratio": 0.1 if self.dataset_size > 500 else 0.2,
                "use_mixed_precision": bool(
                    torch.cuda.is_available() and self.vram_gb >= 6
                ),
            }
        )
        for k in REQUIRED_KEYS:
            cfg.setdefault(k, False if k == "use_mixed_precision" else 0)
        logger.info(
            "추천 설정: batch_size=%d, model_dim=%d, ff_dim=%d, enc_layers=%d, dec_layers=%d",
            cfg["batch_size"],
            cfg["model_dim"],
            cfg["ff_dim"],
            cfg["num_encoder_layers"],
            cfg["num_decoder_layers"],
        )
        return cfg
