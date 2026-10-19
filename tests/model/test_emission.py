"""Test EmissionModel."""

from unittest.mock import patch

import numpy as np

from taurex.cache import OpacityCache


def test_cached_exp_mu_resized_on_wngrid_change():
    """Regression test for cached exp_mu buffers when the grid changes.

    Calling ``model`` on a clipped grid caches ``_cached_exp_mu`` and
    ``_cached_exp_mu2`` at the clipped size. A subsequent call on the full
    native grid used to raise a ``ValueError`` because those buffers were
    never resized.
    """
    from taurex.model import EmissionModel

    with patch.object(OpacityCache, "find_list_of_molecules") as mock_find:
        mock_find.return_value = ["H2O", "CH4"]
        model = EmissionModel(ngauss=4)

    # Avoid needing real opacities by using a manual native grid
    model._native_grid = np.linspace(1000, 5000, 200)

    wn = model.nativeWavenumberGrid
    sub = wn[(wn > 2000) & (wn < 3000)]

    # Cache exp_mu at the clipped size
    clipped, _, _, _ = model.model(wngrid=sub)
    assert model._cached_exp_mu.shape[1] == clipped.shape[0]

    # Full native grid must not raise and must resize the buffers
    native, _, _, _ = model.model(cutoff_grid=False)

    assert native.shape == model._native_grid.shape
    assert model._cached_exp_mu.shape[1] == model._native_grid.shape[0]
    assert model._cached_exp_mu2.shape[1] == model._native_grid.shape[0]
