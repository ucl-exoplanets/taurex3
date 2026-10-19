"""Tests for ExoTransmitOpacity."""

import numpy as np
import pytest

from taurex.opacity.exotransmit import ExoTransmitOpacity


TEMPERATURE_GRID = np.array([300.0, 500.0, 1000.0])
PRESSURE_GRID = np.array([1.0e-4, 1.0e-1, 1.0e2, 1.0e5, 1.0e8])
WAVELENGTHS = np.array([3.0e-6, 5.0e-6, 1.0e-5])  # Metres.


@pytest.fixture
def exo_transmit_path(tmp_path):
    """Create a minimal Exo-Transmit opacity file."""
    lines = [
        " ".join(f"{t:.3f}" for t in TEMPERATURE_GRID),
        " ".join(f"{p:.6e}" for p in PRESSURE_GRID),
    ]

    for wavelength in WAVELENGTHS:
        lines.append(f"{wavelength:.8e}")
        for pressure in PRESSURE_GRID:
            values = np.log10(pressure) + np.arange(TEMPERATURE_GRID.size)
            lines.append(" ".join(f"{v:.6f}" for v in [-4.0, *values]))

    path = tmp_path / "opacH2O.dat"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_exotransmit_pressure_grid_in_pa(exo_transmit_path):
    """Pressures are already in Pa and must not be rescaled (issue #172)."""
    opacity = ExoTransmitOpacity(exo_transmit_path)

    assert np.array_equal(opacity.pressureGrid, PRESSURE_GRID)
