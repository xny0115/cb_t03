"""Automatic hyperparameter tuning utilities."""

from __future__ import annotations

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None
import torch
from ..config import Config


def suggest_config(num_pairs: int) -> dict:
    """Suggest config based on hardware and dataset size."""
    import torch
    if torch.cuda.is_available():
        try:
            vram = torch.cuda.get_device_properties(0).total_memory // 1_073_741_824
        except Exception:
            vram = 0
    else:
        vram = 0
    ram = psutil.virtual_memory().total // 1_073_741_824 if psutil else 0
    cfg = {}
    cfg["d_model"] = 512 if vram >= 8 else 256
    cfg["ff_dim"] = 2048 if vram >= 8 else 1024
    cfg["n_layers"] = 6 if num_pairs > 500 else 4
    cfg["batch_size"] = 32 if vram >= 8 else 8
    cfg["epochs"] = 30 if num_pairs > 300 else 15
    cfg["lr"] = 5e-4 if num_pairs > 300 else 1e-3
    return cfg


class AutoTuner:
    """Estimate reasonable hyperparameters based on hardware and data."""

    def __init__(self, dataset_size: int) -> None:
        self.dataset_size = dataset_size
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        if psutil is not None:
            self.mem_total = psutil.virtual_memory().total // (1024**3)
        else:
            self.mem_total = 0

    def suggest_config(self) -> Config:
        batch_size = 4
        if self.device == "cuda":
            batch_size = min(32, max(4, self.mem_total // 4))
        lr = 3e-4 if self.dataset_size > 1000 else 1e-3
        epochs = 10 if self.dataset_size > 500 else 20
        cfg = Config(
            batch_size=batch_size,
            learning_rate=lr,
            num_epochs=epochs,
        )
        return cfg
