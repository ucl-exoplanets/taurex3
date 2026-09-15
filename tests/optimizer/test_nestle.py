"""Test Nestle optimizer."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from . import BinnedLineObs
from . import LineModel
from . import LineObs
from . import NativeLineModel


class TauLineModel(LineModel):
    """Line model returning a valid optical-depth array for binning."""

    def model(self, wngrid=None, cutoff_grid=True):
        """Run the model and return a concrete tau array."""
        wngrid, flux, _, _ = super().model(wngrid, cutoff_grid)
        return wngrid, flux, np.zeros_like(flux), None


@pytest.mark.slow
@given(m=st.floats(1.0, 2.0), c=st.floats(1.0, 30.0))
@settings(deadline=None)
def test_optimizer(m, c):
    """Test optimizer."""
    from taurex.optimizer import NestleOptimizer

    lm = LineModel()
    lm.m = 1.0
    lm.c = 10.0
    lo = LineObs(m=m, c=c, N=5)
    opt = NestleOptimizer(num_live_points=5, observed=lo, model=lm)
    opt.enable_fit("m")
    opt.enable_fit("c")
    opt.enable_derived("mplusc")
    opt.set_boundary("m", [0.8 * m, 1.2 * m])
    opt.set_boundary("c", [0.8 * c, 1.2 * c])

    opt.fit()

    idx, optimized_map, optimized_median, values = next(opt.get_solution())

    opt.update_model(optimized_map)

    assert lm.m == pytest.approx(m, rel=0.2)
    assert lm.c == pytest.approx(c, rel=0.2)


@pytest.mark.slow
def test_optimizer_fluxbinner():
    """Test retrieval with FluxBinner binning.

    Verifies that the flux-conserving binner works correctly when used
    in the retrieval pipeline: the model is computed on its native fine
    grid, then binned to the observation's coarse grid via FluxBinner.
    """
    np.random.seed(42)
    from taurex.optimizer import NestleOptimizer

    true_m, true_c = 1.5, 15.0

    lm = NativeLineModel()
    lm.m = 1.0
    lm.c = 10.0

    lo = BinnedLineObs(m=true_m, c=true_c, N=10)

    opt = NestleOptimizer(num_live_points=50, observed=lo, model=lm)
    opt.enable_fit("m")
    opt.enable_fit("c")
    opt.set_boundary("m", [0.8 * true_m, 1.2 * true_m])
    opt.set_boundary("c", [0.8 * true_c, 1.2 * true_c])

    opt.fit()

    idx, optimized_map, optimized_median, values = next(opt.get_solution())
    opt.update_model(optimized_map)

    assert lm.m == pytest.approx(true_m, rel=0.2)
    assert lm.c == pytest.approx(true_c, rel=0.2)


@pytest.mark.slow
def test_optimizer_simplebinner():
    """Test retrieval with SimpleBinner binning.

    Verifies that the simple histogram-based binner works correctly
    when used in the retrieval pipeline.
    """
    np.random.seed(42)
    from taurex.binning import SimpleBinner
    from taurex.optimizer import NestleOptimizer

    true_m, true_c = 1.5, 15.0

    lm = NativeLineModel()
    lm.m = 1.0
    lm.c = 10.0

    lo = BinnedLineObs(m=true_m, c=true_c, N=10, binner_cls=SimpleBinner)

    opt = NestleOptimizer(num_live_points=50, observed=lo, model=lm)
    opt.enable_fit("m")
    opt.enable_fit("c")
    opt.set_boundary("m", [0.8 * true_m, 1.2 * true_m])
    opt.set_boundary("c", [0.8 * true_c, 1.2 * true_c])

    opt.fit()

    idx, optimized_map, optimized_median, values = next(opt.get_solution())
    opt.update_model(optimized_map)

    assert lm.m == pytest.approx(true_m, rel=0.2)
    assert lm.c == pytest.approx(true_c, rel=0.2)


@pytest.mark.slow
def test_retrieval_calibration_file_only(tmp_path):
    """Retrieval relying solely on the calibration file.

    ``OffsetSpectraCont`` is constructed with a calibration file but no
    ``broadening_order``, so no ``Broadening_N_k`` parameters are created and
    the binner convolves with the raw line-spread function. The retrieval
    still recovers the model parameters while fitting the per-spectrum
    offsets and error scales.
    """
    np.random.seed(42)

    from taurex.data.spectrum import OffsetSpectraCont
    from taurex.optimizer import NestleOptimizer

    true_m, true_c = 1.5, 15.0

    lm = TauLineModel()
    lm.m = 1.0
    lm.c = 10.0

    # Two coarse observed grids in wavenumber, stored as wavelength (microns).
    wn1 = np.linspace(2500.0, 3330.0, 12)
    wn2 = np.linspace(3400.0, 5000.0, 12)

    def write_spectrum(filename, wn):
        wl = 10000.0 / wn[::-1]
        flux = true_m * wn + true_c
        err = 0.05 * np.abs(flux) + 0.05
        noise = err * np.random.randn(wn.size)
        np.savetxt(filename, np.column_stack([wl, flux[::-1] + noise[::-1], err[::-1]]))

    spec1 = tmp_path / "spec1.dat"
    spec2 = tmp_path / "spec2.dat"
    write_spectrum(spec1, wn1)
    write_spectrum(spec2, wn2)

    # Synthetic calibration file: wavelength (micron) and resolving power.
    cal_wl = np.linspace(1.9, 5.5, 100)
    cal = tmp_path / "calibration.dat"
    np.savetxt(cal, np.column_stack([cal_wl, np.full_like(cal_wl, 100.0)]))

    obs = OffsetSpectraCont(
        path_spectra=[str(spec1), str(spec2)],
        path_broadening=[str(cal), str(cal)],
        broadening_type="stsci_fits",
        wlres=500,
        max_wlbroadening=0.1,
    )

    # Calibration-only mode: the raw profile is used and no broadening
    # parameters are created.
    assert obs.broadening_coeffs is None
    assert not any(k.startswith("Broadening_") for k in obs.fittingParameters)

    opt = NestleOptimizer(num_live_points=50, observed=obs, model=lm)
    opt.enable_fit("m")
    opt.enable_fit("c")
    opt.set_boundary("m", [0.8 * true_m, 1.2 * true_m])
    opt.set_boundary("c", [0.8 * true_c, 1.2 * true_c])

    opt.enable_fit("Offset_1")
    opt.set_boundary("Offset_1", [-1e-2, 1e-2])
    opt.enable_fit("EScale_1")
    opt.set_boundary("EScale_1", [0.8, 5.0])
    opt.enable_fit("Offset_2")
    opt.set_boundary("Offset_2", [-1e-2, 1e-2])
    opt.enable_fit("EScale_2")
    opt.set_boundary("EScale_2", [0.8, 5.0])

    opt.fit()

    idx, optimized_map, optimized_median, values = next(opt.get_solution())
    opt.update_model(optimized_map)

    assert lm.m == pytest.approx(true_m, rel=0.2)
    assert lm.c == pytest.approx(true_c, rel=0.2)
