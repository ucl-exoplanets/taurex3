"""Class for forward spectral modeling."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.binning import Binner
from taurex.core import Citable, DerivedType, Fittable, FittingType
from taurex.log import Logger
from taurex.output import OutputGroup, Writeable
from taurex.types import ModelOutputType

if t.TYPE_CHECKING:
    from taurex.contributions import Contribution
else:
    Contribution = object


class ForwardModel(Logger, Fittable, Writeable, Citable):
    """A base class for producing forward models."""

    def __init__(self, name: str) -> None:
        """Initialise forward model.


        Parameters
        ----------
        name : str
            Name of forward model.

        """
        Logger.__init__(self, name)
        Fittable.__init__(self)
        self.opacity_dict = {}
        self.cia_dict = {}

        self._native_grid = None

        self._derived_parameters = self.derived_parameters()
        self._fitting_parameters = self.fitting_parameters()

        self.contribution_list: t.List[Contribution] = []

    def __getitem__(self, key: str) -> float:
        """Get a fitting parameter value."""
        return self._fitting_parameters[key][2]()

    def __setitem__(self, key: str, value: float) -> None:
        """Set a fitting parameter value."""
        return self._fitting_parameters[key][3](value)

    def defaultBinner(self) -> Binner:  # noqa: N802
        from taurex.binning import NativeBinner

        return NativeBinner()

    def add_contribution(self, contrib: Contribution) -> None:
        """Add a contribution to the forward model."""
        if not isinstance(contrib, Contribution):
            raise TypeError("Is not a a contribution type")
        else:
            if contrib not in self.contribution_list:
                self.contribution_list.append(contrib)
            else:
                raise Exception("Contribution already exists")

    def build(self) -> None:
        """Build forward model."""
        raise NotImplementedError

    def initialize_profiles(self) -> None:
        """Initialize profiles."""
        raise NotImplementedError

    def model(
        self,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        cutoff_grid: t.Optional[bool] = True,
    ) -> ModelOutputType:
        """Computes the forward model for a wngrid"""
        raise NotImplementedError

    def model_contrib(
        self,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        cutoff_grid: t.Optional[bool] = True,
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        t.Dict[
            str,
            t.Tuple[
                npt.NDArray[np.float64], npt.NDArray[np.float64], t.Optional[t.Any]
            ],
        ],
    ]:
        """Computes the forward model for a wngrid with each contribution."""
        raise NotImplementedError

    def model_full_contrib(
        self,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        cutoff_grid: t.Optional[bool] = True,
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        t.Dict[
            str,
            t.List[
                t.Tuple[
                    str,
                    npt.NDArray[np.float64],
                    npt.NDArray[np.float64],
                    t.Optional[t.Any],
                ]
            ],
        ],
    ]:
        """Computes the forward model for a wngrid for each contribution

        Considers each contribution has subcomponents as well.

        """
        raise NotImplementedError

    @property
    def fittingParameters(self) -> t.Dict[str, FittingType]:  # noqa: N802
        """Returns a dictionary of fitting parameters"""
        return self._fitting_parameters

    @property
    def derivedParameters(self) -> t.Dict[str, DerivedType]:  # noqa: N802
        """Returns a dictionary of derived parameters"""
        return self._derived_parameters

    def compute_error(
        self,
        samples: t.Callable[[], float],
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        binner: t.Optional[Binner] = None,
    ) -> t.Tuple[
        t.Dict[str, npt.NDArray[np.float64]], t.Dict[str, npt.NDArray[np.float64]]
    ]:
        """Error of the model and its components given a sample function."""
        return {}, {}

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write forward model to output group."""
        model = output.create_group("ModelParameters")
        model.write_string("model_type", self.__class__.__name__)
        contrib = model.create_group("Contributions")
        for c in self.contribution_list:
            c.write(contrib)

        return model

    def generate_profiles(self) -> t.Dict[str, npt.NDArray[np.float64]]:
        """Generate profiles to store.

        Must return a dictionary of profiles you want to
        store after modeling
        """
        from taurex.util.output import generate_profile_dict

        if hasattr(self, "temperatureProfile"):
            return generate_profile_dict(
                self
            )  # To ensure this change does not break anything
        else:
            return {}

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for forward model."""
        raise NotImplementedError

    def citations(self) -> t.List[str]:
        """Citations for forward model.

        Will either return a list of string or a list of
        :class:`~taurex.data.citation.Citation` objects if
        ``pybtex`` is installed.


        """
        from taurex.core import unique_citations_only

        model_citations = super().citations()
        for c in self.contribution_list:
            model_citations.extend(c.citations())

        return unique_citations_only(model_citations)
