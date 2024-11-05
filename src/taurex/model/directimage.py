"""Direct imaging model."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.constants import PI

from .emission import EmissionModel


class DirectImageModel(EmissionModel):
    """A forward model for direct imaging of exo-planets."""

    def compute_final_flux(
        self, f_total: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute the final flux.

        This is the emission flux that is observed at the telescope directly
        from an exo-planet.

        """
        star_distance_meters = self._star.distance * 3.08567758e16

        sdr = pow((star_distance_meters / 3.08567758e16), 2)
        sdr = 1.0
        planet_radius = self._planet.fullRadius

        return (
            (f_total * (planet_radius**2) * 2.0 * PI)
            / (4 * PI * (star_distance_meters**2))
        ) * sdr

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for this class."""
        return (
            "direct",
            "directimage",
        )
