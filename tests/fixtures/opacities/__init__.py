"""Fixtures for opacities tests."""

import numpy as np
import pytest

from .genopacs import gen_opacity


@pytest.fixture(params=["1H2-O", "H2O"])
def pickle_opac(tmpdir, request):
    """Create a dummy pickle opacity."""
    p = tmpdir.mktemp("opac").join(f"{request.params}_adslkjd.pickle")
    p.dump(gen_opacity(np.linspace(200, 2000, 10), np.logspace(6, 0, 12), 100, "H2O"))
    return p
