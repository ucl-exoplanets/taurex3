"""Direct imaging model."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.constants import PI

from .emission import EmissionModel

def compute_direct_image_final_flux(
    f_total, planet_radius, star_distance
):
    return f_total*(planet_radius**2)/(star_distance**2)


class DirectImageModel(EmissionModel):
    """A forward model for direct imaging of exo-planets."""

    def compute_final_flux(
        self, f_total: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute the final flux.

        This is the emission flux that is observed at the telescope directly
        from an exo-planet.

        """
        return (
            compute_direct_image_final_flux(f_total, self._planet.fullRadius, self._star.distance * 3.08567758e16)
        )

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for this class."""
        return (
            "direct",
            "directimage",
        )
