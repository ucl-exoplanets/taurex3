"""Tests for fixing parameter values from the input file."""

import textwrap

from taurex.optimizer import NestleOptimizer
from taurex.parameter import ParameterParser


def _write_spectrum(path, first_wavelength):
    """Write a minimal three-point spectrum."""
    lines = [f"{first_wavelength + 0.1 * index} 0.01 0.001 0.05" for index in range(3)]
    path.write_text("\n".join(lines) + "\n")


def test_fixed_broadening_coefficient_keeps_its_value(tmp_path):
    """A non-fitted resolution_poly coefficient keeps the value from the file.

    Reproduces a ``resolution_poly`` setup where one coefficient should stay
    fixed instead of being sampled. Before ``:value`` was supported, the
    coefficient kept its hard-coded default and there was no way to set it
    from the input file.
    """
    first = tmp_path / "first.dat"
    second = tmp_path / "second.dat"
    _write_spectrum(first, 1.0)
    _write_spectrum(second, 2.0)

    parfile = tmp_path / "config.par"
    parfile.write_text(
        textwrap.dedent(
            f"""
            [Observation]
            observation = spectra_instr
            path_spectra = {first}, {second}
            path_broadening = function, function
            broadening_basis = resolution_poly
            broadening_order = 1, -1

            [Fitting]
            Broadening_1_0:fit = True
            Broadening_1_0:prior = "Uniform(bounds=(-14, -8))"
            Broadening_1_1:fit = False
            Broadening_1_1:value = 255
            """
        ).strip()
    )

    parser = ParameterParser()
    parser.read(str(parfile))
    observation = parser.generate_observation()

    optimizer = NestleOptimizer(observed=observation)
    parser.setup_optimizer(optimizer)

    fixed = optimizer.avail_fit_parameters["Broadening_1_1"]
    assert observation.broadening_coeffs[0][1] == 255.0
    assert optimizer._binner._broadening_coeffs[0][1] == 255.0
    assert fixed.value == 255.0
    assert fixed.to_fit is False
    assert "Broadening_1_1" not in optimizer.fit_names

    fitted = optimizer.avail_fit_parameters["Broadening_1_0"]
    assert fitted.to_fit is True
    assert fitted.prior.boundaries() == (-14.0, -8.0)
