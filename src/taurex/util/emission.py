"""Functions related to computing emission spectrums."""

import math
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.constants import KBOLTZ, PI, PLANCK, SPDLIGT

# NUMBA Functions
try:
    import numba
    from numba import float64

    @numba.vectorize([float64(float64)], fastmath=True)
    def _convert_lamb(lamb: npt.NDArray[np.float64]) -> npt:
        """Convert wavenumber in :math:`\\mu m` to :math:`m`."""
        return 10000 * 1e-6 / lamb

    @numba.vectorize([float64(float64, float64)], fastmath=True)
    def _black_body_vec(wl: npt.NDArray[np.float64], temp: float):
        """Compute black body spectrum."""
        return (
            (PI * (2.0 * PLANCK * SPDLIGT**2) / (wl) ** 5)
            * (1.0 / (np.exp((PLANCK * SPDLIGT) / (wl * KBOLTZ * temp)) - 1))
            * 1e-6
        )

    @numba.njit(fastmath=True, parallel=False)
    def black_body_numba(lamb: npt.NDArray[np.float64], temp: float):
        """Compute black body spectrum using numba."""
        wl = _convert_lamb(lamb)
        return _black_body_vec(wl, temp)

    @numba.njit(fastmath=True, parallel=False)
    def black_body_numba_II(lamb, temp):  # noqa: N802
        """Compute black body spectrum (alt algo) using numba."""
        n = lamb.shape[0]
        out = np.zeros_like(lamb)
        conversion = 10000 * 1e-6
        # for n in range(N):
        #     wl[n] = 10000*1e-6/lamb[n]

        factor = PI * (2.0 * PLANCK * SPDLIGT**2) * 1e-6 / conversion**5
        c2 = PLANCK * SPDLIGT / (KBOLTZ * temp) / conversion

        for x in range(n):
            out[x] = factor * lamb[x] ** 5 / (math.exp(c2 * lamb[x]) - 1)

        return out

except ImportError:
    print("Numba not installed, using numpy instead")

    def black_body_numba(lamb: npt.NDArray[np.float64], temp: float):
        """Compute black body spectrum using numpy (numba not available)."""
        return black_body_numpy(lamb, temp)

    def black_body_numba_II(lamb: npt.NDArray[np.float64], temp: float):  # noqa: N802
        """Compute black body spectrum using numpy (numba not available)."""
        return black_body_numpy(lamb, temp)


def black_body_numexpr(lamb: npt.NDArray, temp: npt.NDArray) -> npt.NDArray:
    """Compute black body spectrum using numexpr."""
    import numexpr as ne

    wl = ne.evaluate("10000*1e-6/lamb")  # noqa: F841

    return ne.evaluate(
        "(PI* (2.0*PLANCK*SPDLIGT**2)/(wl)**5) * (1.0/(exp((PLANCK * SPDLIGT)"
        " / (wl * KBOLTZ * temp))-1))*1e-6"
    )


def black_body_numpy(lamb: npt.NDArray, temp: npt.NDArray) -> npt.NDArray:
    """Compute black body spectrum using numpy.

    Parameters
    ----------
    lamb : npt.NDArray
        Wavelengths in microns
    temp : npt.NDArray
        Temperature in Kelvin

    Returns
    -------
    npt.NDArray
        Black body spectrum

    """
    h = 6.62606957e-34
    c = 299792458
    k = 1.3806488e-23
    pi = np.pi

    temp = np.atleast_1d(temp)

    lamb = np.atleast_1d(lamb)

    temp_ = temp.ravel()
    lamb_ = lamb.ravel()

    final_shape = temp_.shape + lamb_.shape

    temp_ = np.broadcast_to(temp_[:, None], final_shape)
    lamb_ = np.broadcast_to(lamb_, final_shape)
    wl = 10000 / lamb_
    exponent = np.exp((h * c) / (wl * 1e-6 * k * temp_))
    bb = (pi * (2.0 * h * c**2) / (wl * 1e-6) ** 5) * (1.0 / (exponent - 1))
    final = bb * 1e-6

    if temp.size == 1:
        final = final.squeeze(axis=0)
    else:
        final = final.reshape(temp.shape + (-1,))
    if lamb.size == 1:
        final = final.squeeze(axis=-1)
    elif final.ndim > 1:
        final = final.reshape(final.shape[:-1] + lamb.shape)

    return final


def integrate_emission_layer(
    dtau: npt.NDArray, layer_tau: npt.NDArray, mu: npt.NDArray, bb: npt.NDArray
) -> t.Tuple[npt.NDArray, npt.NDArray]:
    """Integrate emission layer.

    Parameters
    ----------
    dtau : npt.NDArray
        Optical depth of layer
    layer_tau : npt.NDArray
        Optical depth of layer
    mu : npt.NDArray
        Cosine of zenith angle
    BB : npt.NDArray
        Black body spectrum

    Returns
    -------
    t.Tuple[npt.NDArray, npt.NDArray]
        Integrated emission layer, optical depth of layer

    """

    _mu = 1 / mu[:, None]
    _tau = np.exp(-layer_tau) - np.exp(-dtau)

    return bb * (np.exp(-layer_tau * _mu) - np.exp(-dtau * _mu)), _tau


black_body = black_body_numba
"""Black body function to use."""
