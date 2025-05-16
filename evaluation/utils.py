from typing import Tuple, Optional

import numpy as np
import numpy.typing as npt

from colors import green, red


def is_numpy(
    array: np.ndarray,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[npt.DTypeLike] = None,
) -> bool:
    if not isinstance(array, np.ndarray):
        return False

    if shape is not None and array.shape != shape:
        return False

    if dtype is not None and array.dtype != np.dtype(dtype):
        return False

    return True


def is_shape(shape: Tuple[int, ...], ndim: Optional[int] = None) -> bool:
    if not isinstance(shape, tuple):
        return False

    for dimension in shape:
        if not isinstance(dimension, int):
            return False

    if ndim is not None and len(shape) != ndim:
        return False

    return True


def get_icon(b: bool):
    return green("✓") if b else red("✕")
