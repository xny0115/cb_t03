import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.array_demo import numpy_multiply, python_multiply


def test_array_multiply():
    a = [1, 2, 3]
    b = [4, 5, 6]
    assert numpy_multiply(a, b) == [4, 10, 18]
    assert python_multiply(a, b) == [4, 10, 18]
