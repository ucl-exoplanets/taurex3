from taurex.data.spectrum.array import ArraySpectrum
from taurex.data.spectrum.iraclis import IraclisSpectrum
from taurex.data.spectrum.observed import ObservedSpectrum
from taurex.data.spectrum.offsetspectrum import OffsetSpectra
from taurex.data.spectrum.offsetspectrum import OffsetSpectraCont
from taurex.data.spectrum.spectrum import BaseSpectrum


try:
    from taurex.data.spectrum.lightcurve import ObservedLightCurve
except ImportError:
    print("pylightcurve not install. Ignoring")
