"""Optimizer test fixtures."""

import numpy as np

from taurex.core import derivedparam
from taurex.core import fitparam
from taurex.model import ForwardModel
from taurex.spectrum import BaseSpectrum


class LineModel(ForwardModel):
    """Line model for testing optimizers."""

    def __init__(self):
        """Initialize LineModel."""
        super().__init__(self.__class__.__name__)
        self._m = 0.5
        self._c = 10.0

        self._x = np.linspace(1, 100, 50)

    @fitparam(param_name="c")
    def c(self):
        """Line parameter c."""
        return self._c

    @c.setter
    def c(self, value):
        self._c = value

    @fitparam(param_name="m")
    def m(self):
        """Line parameter m."""
        return self._m

    @m.setter
    def m(self, value):
        self._m = value

    def model(self, wngrid=None, cutoff_grid=True):
        """Run the model."""
        if wngrid is None:
            wngrid = self._x
        return wngrid, self._m * wngrid + self._c, None, None

    def build(self):
        """Build the model."""
        pass

    def initialize_profiles(self):
        """Initialize profiles."""
        pass

    @property
    def chemistry(self):
        """Chemistry property."""

        class Dummy:
            """Dummy chemistry."""

            @property
            def muProfile(self):
                """Mu profile."""
                return [1.0]

        test = Dummy()
        return test

    @derivedparam(param_name="mplusc")
    def mplusc(self):
        """Derived parameter m+c."""
        return self._m + self._c


class LineObs(BaseSpectrum):
    """Line observation for testing optimizers."""

    def create_binner(self):
        """Creates the appropriate binning object."""
        from taurex.binning import NativeBinner

        return NativeBinner()

    def __init__(self, m, c, N):
        """Initialize LineObs."""
        super().__init__("LineObs")
        self._m = m
        self._c = c
        self._x = np.linspace(1, 100, N)
        self._y = self._m * self._x + self._c
        self._yerr = 0.1 + 0.1 * np.random.rand(N)
        self._y += self._yerr * np.random.randn(N)

    @property
    def spectrum(self):
        """Spectrum property."""
        return self._y

    @property
    def wavenumberGrid(self):
        """Wavenumber grid property."""
        return self._x

    @property
    def errorBar(self):
        """Error bar property."""
        return self._yerr


class LineObsWithParams(BaseSpectrum):
    """Line observation with params."""

    def create_binner(self):
        """Creates the appropriate binning object."""
        from taurex.binning import NativeBinner

        return NativeBinner()

    def __init__(self, m, c, N):
        """Initialize LineObsWithParams."""
        super().__init__("LineObs")
        self._m = m
        self._c = c
        self._lol = 40
        self._x = np.linspace(1, 100, N)
        self._y = self._m * self._x + self._c
        self._yerr = 0.1 + 0.1 * np.random.rand(N)
        self._y += self._yerr * np.random.randn(N)

    @property
    def spectrum(self):
        """Spectrum property."""
        return self._y

    @property
    def wavenumberGrid(self):
        """Wavenumber grid property."""
        return self._x

    @property
    def errorBar(self):
        """Error bar property."""
        return self._yerr

    @fitparam(param_name="lol")
    def lol(self):
        """LOL parameter."""
        return self._lol

    @lol.setter
    def lol(self, value):
        self._lol = value
