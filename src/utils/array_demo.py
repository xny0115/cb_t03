from __future__ import annotations

"""Array manipulation examples using numpy and pure Python."""

from typing import List
import numpy as np


def numpy_multiply(a: List[int], b: List[int]) -> List[int]:
    """Return element-wise products of two lists using numpy arrays."""
    arr1 = np.array(a)
    arr2 = np.array(b)
    return (arr1 * arr2).tolist()


def python_multiply(a: List[int], b: List[int]) -> List[int]:
    """Return element-wise products of two lists using Python loops."""
    if len(a) != len(b):
        raise ValueError("length mismatch")
    result = []
    for x, y in zip(a, b):
        result.append(x * y)
    return result


if __name__ == "__main__":
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]
    print("numpy:", numpy_multiply(arr1, arr2))
    print("python:", python_multiply(arr1, arr2))
