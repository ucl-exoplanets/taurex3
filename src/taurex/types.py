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
    """Return the default floating-point dtype for computation arrays.

    Returns
    -------
    np.dtype
        ``np.float64``
    """
    return np.float64
