"""Tests for PHOENIX stellar models."""

from unittest.mock import MagicMock
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest

from taurex.stellar import PhoenixStar


def create_fits_file(tp, lg, mtl):
    """Create fits file."""
    temp_string = ("%3.1f" % (tp / 100)).zfill(5)
    temp_logg = "%1.1f" % lg
    temp_mtl = "%1.1f" % mtl
    if lg >= 0:
        temp_logg = f"-{temp_logg}"
    if mtl >= 0:
        temp_mtl = f"+{temp_mtl}"

    final_string = (
        f"lte{temp_string}{temp_logg}a{temp_mtl}.ksajdfhaklsjdhf.spec.fits.gz"
    )

    return final_string


def test_phoenix_find_spectrum(tmpdir):
    """Test phoenix find spectrum."""
    # Suppress spectrum loading
    with patch.multiple(
        "taurex.stellar.PhoenixStar",
        get_avail_phoenix=MagicMock(),
        recompute_spectra=MagicMock(),
    ):
        phoenix = PhoenixStar(phoenix_path=str(tmpdir))

    temp = np.arange(1000, 2000, 100, dtype=np.float64)
    logg = [0.0, 1.0, 2.0, 3.0]
    metal = [-1.0, -2.0, 0.0, 1.0, 2.0]
    file_list = []
    test_cases = []
    for tp in temp:
        for lg in logg:
            for mtl in metal:
                filename = create_fits_file(tp, lg, mtl)
                file_list.append(filename)
                test_cases.append((filename, tp, lg, mtl))

    with patch("glob.glob", return_value=file_list):
        phoenix.get_avail_phoenix()

    # Test if we find the right temperatures, logg and Zs

    assert set(temp) == set(phoenix._T_list)
    assert set(logg) == set(phoenix._Logg_list)
    assert set(metal) == set(phoenix._Z_list)

    # Test if we select the correct file
    for fn, tp, lg, mtl in test_cases:
        phoenix._logg = lg
        phoenix._temperature = tp
        phoenix._metallicity = mtl
        assert phoenix.find_nearest_file() == fn


def test_phoenix_quantity_inputs_and_setters(tmpdir):
    """PHOENIX normalizes constructor values and its overridden setters."""
    with patch.multiple(
        "taurex.stellar.PhoenixStar",
        get_avail_phoenix=MagicMock(),
        recompute_spectra=MagicMock(),
    ):
        phoenix = PhoenixStar(
            temperature=1 * u.kK,
            radius=1 * u.R_earth,
            mass=2 * u.M_earth,
            distance=1 * u.kpc,
            phoenix_path=str(tmpdir),
        )

        assert phoenix.temperature == pytest.approx(1000)
        assert phoenix.radius == pytest.approx(u.R_earth.to(u.m))
        assert phoenix.mass == pytest.approx((2 * u.M_earth).to_value(u.kg))
        assert phoenix.distance == pytest.approx(1000)

        phoenix.temperature = 2000 * u.K
        phoenix.mass = 3 * u.M_earth

        assert phoenix.temperature == pytest.approx(2000)
        assert phoenix.mass == pytest.approx((3 * u.M_earth).to_value(u.kg))
