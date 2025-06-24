import pytest
pytest.importorskip("torch")
from src.utils.logger import setup_logger
import logging
import re
import warnings

pytestmark = pytest.mark.requires_torch


def test_logging_debug_info(tmp_path):
    log_path = setup_logger()
    logger = logging.getLogger('test')
    logger.debug('debug line')
    logger.info('info line')
    text = log_path.read_text(encoding='utf-8')
    assert 'debug line' in text
    assert 'info line' in text
    assert re.match(r"\d{6}_\d{4}\.json$", log_path.name)


def test_warning_filters():
    setup_logger()
    with warnings.catch_warnings(record=True) as records:
        warnings.warn("Possible nested set at position 13", FutureWarning)
        warnings.warn("Torch was not compiled with flash attention", UserWarning)
        assert not records
