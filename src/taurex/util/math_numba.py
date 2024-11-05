"""Math functions for Numba."""
# flake8: noqa
import math

import numba
import numpy as np
from numba import float64, vectorize


@numba.vectorize([float64(float64, float64, float64, float64)])
def _expstage0(x1, x2, x11, x21):
    return x1 * x11 - x2 * (x11 - x21)


@numba.vectorize([float64(float64, float64)], fastmath=True)
def _expstage1(x1, x2):
    return math.log(x1 / x2)


@numba.vectorize([float64(float64, float64)], fastmath=True)
def _expstage2(C, x):
    return C * x


@numba.vectorize([float64(float64, float64, float64)], fastmath=True)
def _expstage3(C, x1, x2):
    return C * x1 * x2


@numba.njit(nogil=True, fastmath=True)
def interp_exp_and_lin_broken(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax):
    res = np.zeros_like(x11)
    x0 = -Pmin
    x1 = Pmax + x0
    x2 = P + x0
    factor1 = 1.0 / (T * (Tmax - Tmin))
    factor2 = 1.0 / x1
    x3 = _expstage0(x1, x2, x11, x21)
    x4 = _expstage0(x1, x2, x12, x22)
    x5 = _expstage1(x3, x4)
    x6 = _expstage2(Tmax * (-T + Tmin) * factor1, x5)
    x7 = _expstage3(factor2, x3, x6)
    for i in range(x11.shape[0]):
        res[i] = x7[i] * math.exp(x6[i])
    return res


@numba.njit
def interp_lin_numba(x11, x12, P, Pmin, Pmax):
    # return (x11*(Pmax - Pmin) - (P - Pmin)*(x11 - x12))/(Pmax - Pmin)
    N = x11.shape[0]
    diff = Pmax - Pmin
    scale = (P - Pmin) / diff
    out = np.zeros_like(x11)
    for n in range(N):
        out[n] = x11[n] - scale * (x11[n] - x12[n])

    return out


@numba.njit
def intepr_bilin_numba_II(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax):
    N = x11.shape[0]
    Pdiff = Pmax - Pmin
    Tdiff = Tmax - Tmin
    Pscale = (P - Pmin) / Pdiff
    Tscale = (T - Tmin) / Tdiff

    out = np.zeros_like(x11)

    for n in range(N):
        out[n] = (
            x11[n]
            - Pscale * (x11[n] - x21[n])
            - Pscale * Tscale * (x21[n] - x11[n] + x12[n] - x22[n])
            - Tscale * (x11[n] - x12[n])
        )

    return out


@numba.vectorize([float64(float64, float64, float64)], fastmath=True)
def _linstage0(x11, x21, x):
    return x * (x11 - x21)


@numba.njit(nogil=True, fastmath=True)
def intepr_bilin_numba(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax):
    x0 = -Pmin
    x1 = Pmax + x0
    x2 = -Tmin
    x3 = Tmax + x2
    x4 = P + x0

    factor = 1.0 / (x1 * x3)

    x5 = _linstage0(x11, x21, x4)
    x6 = _linstage0(x11, x12, x1)
    x7 = _linstage0(x12, x22, x4)
    res = np.zeros_like(x11)
    for i in range(x11.shape[0]):
        res[i] = (
            x1 * x11[i] * x3 - x3 * x5[i] - (T + x2) * (x6[i] + x7[i] - x5[i])
        ) * factor

    return res


@numba.njit(fastmath=True)
def interp_exp_numba(x11, x12, temperature, temperature_min, temperature_max):
    N = x11.shape[0]
    scale = (
        temperature_max
        * (temperature_min - temperature)
        / (temperature_max - temperature_min)
    )
    out = np.zeros_like(x11)
    for n in range(N):
        out[n] = x11[n] * np.exp(scale * np.log(x11[n] / x12[n]))

    return out
