"""Tests for observed binning overrides in the parameter parser."""

import textwrap

from taurex.binning import FluxBinnerConv
from taurex.instruments import InstrumentFile
from taurex.parameter import ParameterParser


def test_observed_binning_overrides_are_reused_by_instrument(tmp_path):
    """Check that binning overrides are applied per instrument."""
    first = tmp_path / "spec_1.dat"
    first.write_text("1.0 0.01 0.001 0.05\n1.2 0.02 0.001 0.05\n")

    second = tmp_path / "spec_2.dat"
    second.write_text("2.0 0.03 0.002 0.10\n2.3 0.04 0.002 0.10\n")

    instrument_file = tmp_path / "instrument_noise.dat"
    instrument_file.write_text("1.0 1e-4 0.05\n1.2 1e-4 0.05\n")

    parfile = tmp_path / "config.par"
    parfile.write_text(
        textwrap.dedent(
            f"""
            [Observation]
            observation = spectra_w_offsets
            path_spectra = {first}, {second}

            [Binning]
            bin_type = observed
            broadening_type = none
            wlshift = 0.01, -0.02

            [Instrument]
            instrument = file
            filename = {instrument_file}
            """
        ).strip()
    )

    parser = ParameterParser()
    parser.read(str(parfile))

    observation = parser.generate_observation()
    binning, _ = parser.generate_binning(observation=observation)
    instrument, _ = parser.generate_instrument(binner=binning)

    assert isinstance(binning, FluxBinnerConv)
    assert isinstance(instrument, InstrumentFile)
    assert instrument._binner is binning
