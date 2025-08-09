from __future__ import annotations

"""Application entrypoint for the local webview."""

import platform
import warnings

try:  # pragma: no cover - optional dependency
    import webview  # type: ignore
except Exception:  # pragma: no cover
    import time

    class _DummyWebview:
        def create_window(self, *args, **kwargs) -> None:
            pass

        def start(self, *args, **kwargs) -> None:
            while True:
                time.sleep(1)

    webview = _DummyWebview()

from src.utils.logger import setup_logger
import argparse
import os
from pathlib import Path


def _ensure_cuda_torch() -> None:
    """Install CUDA-enabled torch package if needed."""
    import importlib
    import subprocess
    import sys
    import torch

    if torch.cuda.is_available():
        return
    url = (
        "https://download.pytorch.org/whl/cu118/torch-2.3.0%2Bcu118-cp310-cp310-win_amd64.whl"
    )
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", url])
        importlib.reload(torch)
        assert torch.cuda.is_available(), "CUDA torch install failed"
    except Exception as exc:  # pragma: no cover - best effort
        print(f"CUDA torch install failed: {exc}")


def main() -> None:
    """Entry point for training or serving."""
    parser = argparse.ArgumentParser(description="Chatbot runner")
    parser.add_argument(
        "--mode",
        choices=["train", "pretrain", "serve"],
        default="serve",
    )
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--data-dir")
    parser.add_argument("--model-path")
    args = parser.parse_args()

    setup_logger()
    import os
    if os.environ.get("SKIP_CUDA_INSTALL") != "1":
        _ensure_cuda_torch()

    if args.config_path:
        os.environ["CONFIG_PATH"] = args.config_path
    if args.data_dir:
        os.environ["DATA_DIR"] = args.data_dir
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path


    if args.mode == "pretrain":
        from src.service.service import ChatbotService

        svc = ChatbotService()
        svc.start_training("pretrain")
        return
    elif args.mode == "train":
        from src.service.service import ChatbotService

        svc = ChatbotService()
        svc.start_training("finetune")
        return

    from src.service.service import ChatbotService
    from src.ui.backend import WebBackend
    svc = ChatbotService()
    api = WebBackend(svc)
    webview.create_window("Chatbot", "ui.html", js_api=api)
    webview.start(gui="edgechromium", http_server=False)


if __name__ == "__main__":
    main()
