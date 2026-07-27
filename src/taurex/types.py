"""Types for TauREx 3."""

import os
import pathlib
import typing as t

import numpy as np
import numpy.typing as npt


T = t.TypeVar("T")

ModelOutputType = t.Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    t.Union[npt.NDArray[np.float64], None],
    t.Union[t.Dict, T, None],
]
"""Model output type."""

ScalarType = t.Union[float, int, np.float64, np.int64]
"""Scalar type."""

ArrayType = t.Union[np.ndarray, npt.ArrayLike]
"""Array type."""

AnyValType = t.Union[ScalarType, ArrayType]
"""Any value type."""

PathLike = t.Union[str, bytes, os.PathLike, pathlib.Path]
"""Path like type."""


def get_float_dtype():
    """Return the floating-point dtype based on the global float32_mode setting.

    When ``float32_mode`` is set to ``True`` in the ``[Global]`` section of
    the parameter file, all arrays use ``np.float32`` instead of the default
    ``np.float64``. This can significantly reduce memory usage and may improve
    I/O performance.

    Returns
    -------
    np.dtype
        ``np.float32`` if ``GlobalCache()["float32_mode"]`` is truthy,
        ``np.float64`` otherwise.
    """
    from taurex.cache import GlobalCache

    if GlobalCache()["float32_mode"]:
        return np.float32
    return np.float64
