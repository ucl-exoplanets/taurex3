"""Test multinest optimizer."""

import numpy as np

from taurex.optimizer.multinest import MultiNestOptimizer
from taurex.optimizer.multinest import read_mode_chains

from . import LineModel
from . import LineObs


def _sample(weight, params):
    """Create a single multinest posterior sample."""
    return [weight, -2.0 * np.log(weight), *params]


def _write_separate_file(path, modes, blanks=2, skip_modes=()):
    """Write a multinest ``post_separate.dat`` file.

    ``modes`` is a sequence of modes, where each mode is a sequence of samples.
    Modes listed in ``skip_modes`` are written without any sample.
    """
    lines = []
    for idx, mode in enumerate(modes):
        lines.extend([""] * blanks)
        if idx in skip_modes:
            # A mode without usable samples only writes the separators
            continue
        for sample in mode:
            lines.append("".join(f"{value:28.6E}" for value in sample))
    path.write_text("\n".join(lines) + "\n")


def test_multimode_default_off(tmp_path):
    """Test that multiple modes are not searched for by default."""
    opt = MultiNestOptimizer(
        multi_nest_path=str(tmp_path),
        observed=LineObs(m=1.0, c=2.0, N=10),
        model=LineModel(),
    )

    assert opt.multimodes is False


def test_read_mode_chains_single_mode(tmp_path):
    """Test reading a file that stores a single mode."""
    sep_file = tmp_path / "1-post_separate.dat"
    _write_separate_file(
        sep_file,
        [[_sample(0.5, [1.0, 2.0]), _sample(0.5, [3.0, 4.0])]],
    )

    samples, weights = read_mode_chains(sep_file)

    assert len(samples) == 1
    assert len(weights) == 1
    assert np.allclose(samples[0], [[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(weights[0], [0.5, 0.5])


def test_read_mode_chains_multiple_modes(tmp_path):
    """Test that modes are separated by any number of blank lines."""
    modes = [
        [_sample(0.5, [1.0, 2.0]), _sample(0.5, [3.0, 4.0])],
        [_sample(0.25, [5.0, 6.0]), _sample(0.75, [7.0, 8.0])],
        [_sample(1.0, [9.0, 10.0])],
    ]

    for blanks in (1, 2, 3):
        sep_file = tmp_path / f"1-post_separate_{blanks}.dat"
        _write_separate_file(sep_file, modes, blanks=blanks)

        samples, weights = read_mode_chains(sep_file)

        assert len(samples) == len(modes)
        assert len(weights) == len(modes)
        assert np.allclose(samples[1], [[5.0, 6.0], [7.0, 8.0]])
        assert np.allclose(weights[2], [1.0])


def test_read_mode_chains_ignores_empty_modes(tmp_path):
    """Test that modes without samples are not stored."""
    sep_file = tmp_path / "1-post_separate.dat"
    _write_separate_file(
        sep_file,
        [
            [_sample(0.5, [1.0, 2.0])],
            [_sample(0.5, [3.0, 4.0])],
            [_sample(1.0, [5.0, 6.0])],
        ],
        skip_modes=(0, 2),
    )

    samples, weights = read_mode_chains(sep_file)

    assert len(samples) == 1
    assert np.allclose(samples[0], [[3.0, 4.0]])
    assert np.allclose(weights[0], [0.5])
