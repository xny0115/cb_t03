from __future__ import annotations

import logging
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

def save_checkpoint(state: dict, path: Path):
    """Saves a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info("Checkpoint saved to %s", path)

def load_checkpoint(path: Path, model, optimizer, scheduler, scaler) -> dict:
    """Loads a training checkpoint from the given path."""
    if not path.exists():
        logger.warning("Checkpoint file not found: %s", path)
        return {}

    logger.info("Loading checkpoint from %s", path)
    checkpoint = torch.load(path, map_location='cuda')

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning("Checkpoint does not contain 'model_state_dict'.")

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    logger.info("Checkpoint loaded successfully.")
    return checkpoint
