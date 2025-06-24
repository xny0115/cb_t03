import shutil
import pytest
import os
import importlib.util


@pytest.fixture(autouse=True)
def _force_small(monkeypatch):
    monkeypatch.setenv("DATASET_MODE", "small")

def pytest_collection_modifyitems(config, items):
    if shutil.which("node") is None:
        skip = pytest.mark.skip(reason="Node.js not present")
        for item in items:
            if "requires_node" in item.keywords:
                item.add_marker(skip)
    for item in items:
        if _lib_missing("torch") or _lib_missing("soynlp"):
            skip_dl = pytest.mark.skip(reason="torch/soynlp not installed in CI")
            if "requires_torch" in item.keywords:
                item.add_marker(skip_dl)


def _lib_missing(pkg):
    return importlib.util.find_spec(pkg) is None
