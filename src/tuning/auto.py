"""Auto tuning utilities."""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None

logger = logging.getLogger(__name__)


class AutoTuner:
    """Suggest config based on dataset size and hardware."""

    def __init__(self, dataset_size: int) -> None:
        self.dataset_size = dataset_size
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
        """Return recommended hyperparameters."""
        cfg: Dict[str, Any] = {}
        high_mem = self.vram_gb >= 8 or self.ram_gb >= 16
        cfg["batch_size"] = 32 if high_mem else 8
        cfg["model_dim"] = 512 if self.vram_gb >= 8 else 256
        cfg["ff_dim"] = 2048 if cfg["model_dim"] >= 512 else 1024
        cfg["num_encoder_layers"] = 6 if self.dataset_size > 500 else 4
        cfg["num_decoder_layers"] = cfg["num_encoder_layers"]
        cfg["num_epochs"] = 30 if self.dataset_size > 300 else 15
        cfg["learning_rate"] = 5e-4 if self.dataset_size > 300 else 1e-3
        return cfg
