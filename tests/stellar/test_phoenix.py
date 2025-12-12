"""Tests for PHOENIX stellar models."""
from unittest.mock import MagicMock, patch

import numpy as np

from taurex.stellar import PhoenixStar


def create_fits_file(tp, lg, mtl):
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
