import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter
from pathlib import Path
from datetime import datetime
import warnings

LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / f"{datetime.now():%y%m%d_%H%M}.log"

class TextFormatter(Formatter):
    """간단한 텍스트 로거 포맷."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        message = record.getMessage()
        if record.exc_info:
            message += f" | {self.formatException(record.exc_info)}"
        return f"[{record.levelname}] {message}"


def setup_logger() -> Path:
    """루트 로거를 구성한다."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", message="Possible nested set", module="soynlp.tokenizer")
    logging.captureWarnings(True)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    formatter = TextFormatter()
    handler.setFormatter(formatter)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(handler)
    root.addHandler(stream)
    return LOG_PATH
