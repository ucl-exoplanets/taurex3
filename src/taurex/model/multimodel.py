"""Composite forward models built from multiple 1D submodels."""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex.binning import Binner
from taurex.chemistry import Chemistry
from taurex.constants import AMU
from taurex.constants import PI
from taurex.core import derivedparam
from taurex.exceptions import InvalidModelException
from taurex.planet import Planet
from taurex.pressure import PressureProfile
from taurex.pressure import SimplePressureProfile
from taurex.stellar import BlackbodyStar
from taurex.stellar import Star
from taurex.temperature import TemperatureProfile
from taurex.util import clip_native_to_wngrid
from taurex.util.emission import black_body

from .directimage import DirectImageModel
from .directimage import compute_direct_image_final_flux
from .emission import EmissionModel
from .emission import contribute_ktau_emission
from .model import ForwardModel
from .transmission import TransmissionModel

if t.TYPE_CHECKING:
    from taurex.contributions import Contribution
    from taurex.output import OutputGroup
    from taurex.spectrum import BaseSpectrum
else:
    Contribution = object
    OutputGroup = object
    BaseSpectrum = object


class InvalidMultiModelException(InvalidModelException):
    """Raised when a composite model has invalid region weights."""


class MultiChemistry:
    """Lightweight chemistry view over multiple submodel chemistry profiles."""

    def __init__(
        self,
        actives: t.Sequence[npt.NDArray[np.float64]],
        inactives: t.Sequence[npt.NDArray[np.float64]],
        mu: t.Sequence[npt.NDArray[np.float64]],
        active_species: t.Optional[t.Sequence[str]] = None,
    ) -> None:
        self._active_profiles = list(actives)
        self._inactive_profiles = list(inactives)
        self._mus = list(mu)
        self._active_species = list(active_species or [])

    @property
    def activeGases(self) -> t.List[str]:  # noqa: N802
        return self._active_species

    @property
    def activeGasMixProfile(self) -> t.Dict[str, npt.NDArray[np.float64]]:  # noqa: N802
        return {
            f"region{index}": profile
            for index, profile in enumerate(self._active_profiles)
        }

    @property
    def inactiveGasMixProfile(self) -> t.Dict[str, npt.NDArray[np.float64]]:  # noqa: N802
        return {
            f"region{index}": profile
            for index, profile in enumerate(self._inactive_profiles)
        }

    @property
    def muProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return np.mean(np.array(self._mus), axis=0)

    @derivedparam(param_name="mu", param_latex="$\\mu$", compute=True)
    def mu(self) -> float:
        return self.muProfile[0] / AMU

    @property
    def hasCondensates(self) -> bool:  # noqa: N802
        return False


class MultiTransitModel(ForwardModel):
    """Weighted combination of multiple transmission-style submodels."""

    def __init__(
        self,
        temperature_profiles: t.Optional[t.Sequence[t.Optional[TemperatureProfile]]] = None,
        chemistry: t.Optional[t.Sequence[t.Optional[Chemistry]]] = None,
        pressure_min: t.Optional[t.Union[float, t.Sequence[float]]] = None,
        pressure_max: t.Optional[t.Union[float, t.Sequence[float]]] = None,
        nlayers: t.Optional[t.Union[int, t.Sequence[int]]] = None,
        pressure_profile: t.Optional[t.Sequence[t.Optional[PressureProfile]]] = None,
        planet: t.Optional[Planet] = None,
        star: t.Optional[Star] = None,
        observation: t.Optional[BaseSpectrum] = None,
        contributions: t.Optional[t.Sequence[t.Optional[t.Sequence[Contribution]]]] = None,
        fractions: t.Optional[t.Sequence[float]] = None,
        use_cuda: bool = False,
    ) -> None:
        super().__init__(self.__class__.__name__)

        region_count = self._determine_region_count(
            temperature_profiles,
            chemistry,
            pressure_profile,
            contributions,
        )

        self._use_cuda = use_cuda
        self._observation = observation
        self._planet = planet or Planet()
        self._star = star or BlackbodyStar()
        self._fractions, self.autofrac = self._normalize_fractions(
            fractions, region_count
        )

        self._temperature_profiles = self._normalize_sequence(
            temperature_profiles, region_count, None
        )
        self._chemistry_profiles = self._normalize_sequence(
            chemistry, region_count, None
        )
        self._pressure_profiles = self._normalize_sequence(
            pressure_profile, region_count, None
        )
        self._pressure_min = self._normalize_scalar_or_sequence(
            pressure_min, region_count, 1e-6
        )
        self._pressure_max = self._normalize_scalar_or_sequence(
            pressure_max, region_count, 1e6
        )
        self._nlayers = self._normalize_scalar_or_sequence(nlayers, region_count, 100)
        self._contribution_sets = self._normalize_contributions(contributions, region_count)

        self.activeGases: t.List[str] = []
        self._active_chems: t.List[npt.NDArray[np.float64]] = []
        self._inactive_chems: t.List[npt.NDArray[np.float64]] = []
        self._mus: t.List[npt.NDArray[np.float64]] = []
        self._temperatures: t.List[npt.NDArray[np.float64]] = []
        self._pressures: t.List[npt.NDArray[np.float64]] = []

        self.tpsClasses: t.List[TemperatureProfile] = []
        self.chemsClasses: t.List[Chemistry] = []
        self.pressClasses: t.List[PressureProfile] = []
        self.contrClasses: t.List[t.List[Contribution]] = []

        self._sub_models = self.create_models()

    @staticmethod
    def _determine_region_count(*groups: t.Any) -> int:
        lengths = [len(group) for group in groups if group is not None]
        if not lengths:
            return 1
        first = lengths[0]
        for length in lengths[1:]:
            if length not in (first, first - 1, first + 1):
                raise ValueError("Inconsistent region counts for multimodel inputs")
        return max(lengths)

    @staticmethod
    def _normalize_sequence(
        value: t.Optional[t.Sequence[t.Any]], count: int, default: t.Any
    ) -> t.List[t.Any]:
        if value is None:
            return [default] * count
        normalized = list(value)
        if len(normalized) != count:
            raise ValueError("Multimodel input lengths must match the number of regions")
        return normalized

    @staticmethod
    def _normalize_scalar_or_sequence(
        value: t.Optional[t.Union[int, float, t.Sequence[t.Union[int, float]]]],
        count: int,
        default: t.Union[int, float],
    ) -> t.List[t.Union[int, float]]:
        if value is None:
            return [default] * count
        if isinstance(value, (int, float)):
            return [value] * count
        normalized = list(value)
        if len(normalized) != count:
            raise ValueError("Multimodel scalar inputs must match the number of regions")
        return normalized

    @staticmethod
    def _normalize_contributions(
        contributions: t.Optional[t.Sequence[t.Optional[t.Sequence[Contribution]]]],
        count: int,
    ) -> t.List[t.List[Contribution]]:
        if contributions is None:
            return [[] for _ in range(count)]
        if len(contributions) != count:
            raise ValueError("Contribution lists must match the number of regions")
        return [list(contrib or []) for contrib in contributions]

    @staticmethod
    def _normalize_fractions(
        fractions: t.Optional[t.Sequence[float]], count: int
    ) -> t.Tuple[t.List[float], bool]:
        if count < 1:
            raise ValueError("Multimodel must have at least one region")
        if fractions is None:
            return [1.0 / count] * count, False

        normalized = list(fractions)
        if len(normalized) == count:
            return normalized, False
        if len(normalized) == count - 1 and count > 1:
            return normalized + [1.0 - sum(normalized)], True
        raise ValueError(
            "fractions must contain either N values or N-1 values for auto-normalization"
        )

    @property
    def chemistry(self) -> MultiChemistry:
        return MultiChemistry(
            self._active_chems,
            self._inactive_chems,
            self._mus,
            active_species=self.activeGases,
        )

    def initialize_profiles(self) -> None:
        self.activeGases = []
        self._active_chems = []
        self._inactive_chems = []
        self._mus = []
        self._temperatures = []
        self._pressures = []
        for model in self._sub_models:
            model.initialize_profiles()
            for gas in model.chemistry.activeGases:
                if gas not in self.activeGases:
                    self.activeGases.append(gas)
            self._active_chems.append(model.chemistry.activeGasMixProfile.copy())
            self._inactive_chems.append(model.chemistry.inactiveGasMixProfile.copy())
            self._mus.append(model.chemistry.muProfile.copy())
            self._temperatures.append(model.temperatureProfile.copy())
            self._pressures.append(model.pressureProfile.copy())

    @staticmethod
    def change_fit_values(value: t.Tuple[t.Any, ...], prefix: str) -> t.Tuple[t.Any, ...]:
        name, latex, fget, fset, mode, to_fit, bounds = value
        return f"{prefix}_{name}", f"{prefix}_{latex}", fget, fset, mode, to_fit, bounds

    def create_models(self) -> t.List[ForwardModel]:
        sub_models: t.List[ForwardModel] = []
        for index in range(len(self._fractions)):
            pressure = self._pressure_profiles[index]
            if pressure is None:
                pressure = SimplePressureProfile(
                    int(self._nlayers[index]),
                    float(self._pressure_min[index]),
                    float(self._pressure_max[index]),
                )
            model = self.create_single_model(
                temperature_profile=self._temperature_profiles[index],
                chemistry=self._chemistry_profiles[index],
                pressure_profile=pressure,
                contributions=self._contribution_sets[index],
            )
            sub_models.append(model)
        return sub_models

    def create_single_model(
        self,
        temperature_profile: t.Optional[TemperatureProfile],
        chemistry: t.Optional[Chemistry],
        pressure_profile: PressureProfile,
        contributions: t.Sequence[Contribution],
    ) -> ForwardModel:
        model_class = self.select_model()
        model = model_class(
            planet=self._planet,
            star=self._star,
            pressure_profile=pressure_profile,
            temperature_profile=temperature_profile,
            chemistry=chemistry,
            contributions=list(contributions),
        )
        self.tpsClasses.append(model.temperature)
        self.chemsClasses.append(model.chemistry)
        self.pressClasses.append(model.pressure)
        self.contrClasses.append(list(model.contribution_list))
        return model

    def generate_auto_factors(self) -> None:
        fraction_bounds = (0.0, 1.0)
        fit_count = len(self._fractions) - 1 if self.autofrac else len(self._fractions)
        for index in range(fit_count):
            def read_fraction(self, index=index):
                return self._fractions[index]

            def write_fraction(self, value, index=index):
                self._fractions[index] = value

            self.add_fittable_param(
                f"fr_{index + 1}",
                f"$fr_{index + 1}$",
                read_fraction,
                write_fraction,
                "linear",
                False,
                fraction_bounds,
            )

    def select_model(self) -> t.Type[ForwardModel]:
        model_class: t.Type[ForwardModel] = TransmissionModel
        if self._use_cuda:
            try:
                from taurex_cuda import TransmissionCudaModel

                model_class = TransmissionCudaModel
            except (ModuleNotFoundError, ImportError):
                self.warning("Cuda plugin not found or not working")
        return model_class

    @property
    def nLayers(self) -> int:  # noqa: N802
        return self._sub_models[0].nLayers

    @property
    def densityProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return np.stack([model.densityProfile for model in self._sub_models])

    @property
    def altitudeProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return np.stack([model.altitudeProfile for model in self._sub_models])

    @property
    def temperatureProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return np.array(self._temperatures)

    @property
    def pressureProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return np.array(self._pressures)

    @property
    def scaleheight_profile(self) -> npt.NDArray[np.float64]:
        return np.stack([model.scaleheight_profile for model in self._sub_models])

    @property
    def gravity_profile(self) -> npt.NDArray[np.float64]:
        return np.stack([model.gravity_profile for model in self._sub_models])

    def determine_coupled_fitting(
        self, profiles: t.Sequence[t.Any]
    ) -> t.List[t.Dict[str, t.Tuple[t.Any, ...]]]:
        coupling = [[profile is target for profile in profiles] for target in profiles]
        fit_params = [{} for _ in range(len(profiles) + 1)]

        if coupling and all(coupling[0]):
            fit_params[-1].update(profiles[0].fitting_parameters())
            return fit_params

        for index, duplicate_mask in enumerate(coupling):
            if any(duplicate_mask[:index]):
                continue
            if any(duplicate_mask[index + 1 :]):
                fit_params[-1].update(profiles[index].fitting_parameters())
            else:
                fit_params[index].update(profiles[index].fitting_parameters())
        return fit_params

    def determine_coupled_contributions(
        self, contributions: t.Sequence[t.Sequence[Contribution]]
    ) -> t.List[t.Dict[str, t.Tuple[t.Any, ...]]]:
        seen: t.List[Contribution] = []
        fit_params = [{} for _ in range(len(contributions) + 1)]

        def contains(obj: Contribution, values: t.Sequence[Contribution]) -> bool:
            return any(obj is value for value in values)

        for index, region_contributions in enumerate(contributions):
            remaining = contributions[index + 1 :]
            for contribution in region_contributions:
                if contains(contribution, seen):
                    continue
                seen.append(contribution)
                if any(contains(contribution, group) for group in remaining):
                    fit_params[-1].update(contribution.fitting_parameters())
                else:
                    fit_params[index].update(contribution.fitting_parameters())
        return fit_params

    @property
    def muProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return np.mean(np.array(self._mus), axis=0)

    @derivedparam(param_name="mu", param_latex="$\\mu$", compute=True)
    def mu(self) -> float:
        return self.muProfile[0] / AMU

    def collect_base_derived_params(self) -> None:
        self._derived_parameters = {}
        self._derived_parameters.update(self.derived_parameters())
        self._derived_parameters.update(self._planet.derived_parameters())
        self._derived_parameters.update(self._star.derived_parameters())

    def collect_base_fitting_params(self) -> None:
        self._fitting_parameters = {}
        self._fitting_parameters.update(self.fitting_parameters())
        self._fitting_parameters.update(self._planet.fitting_parameters())
        self._fitting_parameters.update(self._star.fitting_parameters())

        for fits in (
            self.determine_coupled_fitting(self.tpsClasses),
            self.determine_coupled_fitting(self.chemsClasses),
            self.determine_coupled_fitting(self.pressClasses),
        ):
            final_index = len(fits) - 1
            for index, parameter_group in enumerate(fits):
                for key, value in parameter_group.items():
                    if index == final_index:
                        self._fitting_parameters[key] = value
                    else:
                        prefix = f"m{index + 1}"
                        self._fitting_parameters[f"{prefix}_{key}"] = self.change_fit_values(
                            value, prefix
                        )

        contribution_fits = self.determine_coupled_contributions(self.contrClasses)
        final_index = len(contribution_fits) - 1
        for index, parameter_group in enumerate(contribution_fits):
            for key, value in parameter_group.items():
                if index == final_index:
                    self._fitting_parameters[key] = value
                else:
                    prefix = f"m{index + 1}"
                    self._fitting_parameters[f"{prefix}_{key}"] = self.change_fit_values(
                        value, prefix
                    )

    def build(self) -> None:
        for model in self._sub_models:
            model.build()
        self.initialize_profiles()
        self.generate_auto_factors()
        self.collect_base_fitting_params()
        self.collect_base_derived_params()

    @property
    def nativeWavenumberGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return self._sub_models[0].nativeWavenumberGrid

    def check_exceptions(self) -> None:
        if np.sum(self._fractions) > 1.0 + 1e-12:
            raise InvalidMultiModelException("Sum of fractions cannot exceed 1")
        if np.any(np.array(self._fractions) < 0.0):
            raise InvalidMultiModelException("Fractions cannot be negative")

    def model(
        self,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        cutoff_grid: t.Optional[bool] = True,
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        t.Optional[npt.NDArray[np.float64]],
        t.Optional[t.Any],
    ]:
        native_grid = self.nativeWavenumberGrid
        if wngrid is not None and cutoff_grid:
            native_grid = clip_native_to_wngrid(native_grid, wngrid)

        if self.autofrac:
            self._fractions[-1] = 1.0 - np.sum(self._fractions[:-1])
        self.check_exceptions()
        self.initialize_profiles()

        native_fluxes: t.List[npt.NDArray[np.float64]] = []
        native_taus: t.List[npt.NDArray[np.float64]] = []

        for index, model in enumerate(self._sub_models):
            model_output = model.model(wngrid=native_grid, cutoff_grid=cutoff_grid)
            self._active_chems[index] = model.chemistry.activeGasMixProfile.copy()
            self._inactive_chems[index] = model.chemistry.inactiveGasMixProfile.copy()
            self._mus[index] = model.chemistry.muProfile.copy()
            self._temperatures[index] = model.temperatureProfile.copy()
            native_fluxes.append(model_output[1])
            if model_output[2] is not None:
                native_taus.append(model_output[2])

        final_flux = self.compute_final_flux(native_fluxes)
        final_tau = None
        if native_taus:
            final_tau = np.average(np.array(native_taus), axis=0, weights=self._fractions)
        return native_grid, final_flux, final_tau, None

    def compute_final_flux(
        self, native_fluxes: t.Sequence[npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        return np.sum(np.array(native_fluxes) * np.array(self._fractions)[:, None], axis=0)

    def compute_error(
        self,
        samples: t.Callable[[], t.Iterable[float]],
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        binner: t.Optional[Binner] = None,
    ) -> t.Tuple[t.Dict[str, npt.NDArray[np.float64]], t.Dict[str, npt.NDArray[np.float64]]]:
        from taurex.util.math import OnlineVariance

        tp_profiles = OnlineVariance()
        active_variances = [OnlineVariance() for _ in self._sub_models]
        inactive_variances = [OnlineVariance() for _ in self._sub_models]
        binned_spectrum = OnlineVariance() if binner is not None else None
        native_spectrum = OnlineVariance()

        for weight in samples():
            result = self.model(wngrid=wngrid)
            tp_profiles.update(self.temperatureProfile, weight=weight)
            chemistry = self.chemistry
            for index in range(len(self._sub_models)):
                active_variances[index].update(
                    chemistry.activeGasMixProfile[f"region{index}"], weight=weight
                )
                inactive_variances[index].update(
                    chemistry.inactiveGasMixProfile[f"region{index}"], weight=weight
                )
            native_spectrum.update(result[1], weight=weight)
            if binned_spectrum is not None:
                binned_spectrum.update(binner.bin_model(result)[1], weight=weight)

        profile_dict: t.Dict[str, t.Any] = {
            "temp_profile_std": np.sqrt(tp_profiles.parallelVariance()),
            "active_mix_profile_std": {},
            "inactive_mix_profile_std": {},
        }
        spectrum_dict = {"native_std": np.sqrt(native_spectrum.parallelVariance())}

        for index in range(len(self._sub_models)):
            profile_dict["active_mix_profile_std"][f"m{index + 1}"] = np.sqrt(
                active_variances[index].parallelVariance()
            )
            profile_dict["inactive_mix_profile_std"][f"m{index + 1}"] = np.sqrt(
                inactive_variances[index].parallelVariance()
            )

        if binned_spectrum is not None:
            spectrum_dict["binned_std"] = np.sqrt(binned_spectrum.parallelVariance())

        return profile_dict, spectrum_dict

    def write(self, output: OutputGroup) -> OutputGroup:
        self.model(self.nativeWavenumberGrid)
        model = super().write(output)
        self._planet.write(model)
        self._star.write(model)
        for index, sub_model in enumerate(self._sub_models):
            region_group = model.create_group(f"region+{index}")
            sub_model.chemistry.write(region_group)
            sub_model.temperature.write(region_group)
            sub_model.pressure.write(region_group)
        return model

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("multitransit",)


class MultiEclipseModel(MultiTransitModel):
    """Weighted combination of multiple emission submodels."""

    def __init__(self, *args: t.Any, radius_scaling: bool = False, **kwargs: t.Any) -> None:
        self._radius_scaling = radius_scaling
        super().__init__(*args, **kwargs)

    def select_model(self) -> t.Type[ForwardModel]:
        model_class: t.Type[ForwardModel] = EmissionModel
        if self._use_cuda:
            try:
                from taurex_cuda import EmissionCudaModel

                model_class = EmissionCudaModel
            except (ModuleNotFoundError, ImportError):
                self.warning("Cuda plugin not found or not working")
        if self._radius_scaling:
            model_class = EmissionModelRadiusScale
        return model_class

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("multieclipse",)


class EmissionModelRadiusScale(EmissionModel):
    """Emission model variant that scales each layer by its radius."""

    def evaluate_emission_ktables(
        self, wngrid: npt.NDArray[np.float64], return_contrib: bool
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        from taurex.contributions import AbsorptionContribution
        from taurex.util import compute_dz

        dz = compute_dz(self.altitudeProfile)
        total_layers = self.nLayers
        density = self.densityProfile
        wngrid_size = wngrid.shape[0]
        temperature = self.temperatureProfile
        tau = np.zeros(shape=(self.nLayers, wngrid_size))
        surface_tau = np.zeros(shape=(1, wngrid_size))
        layer_tau = np.zeros(shape=(1, wngrid_size))
        tmp_tau = np.zeros(shape=(1, wngrid_size))
        dtau = np.zeros(shape=(1, wngrid_size))
        planet_radius = self._planet.fullRadius
        layer_radius = self.altitudeProfile + self._planet.fullRadius

        mol_type = AbsorptionContribution
        non_molecular = [
            contribution
            for contribution in self.contribution_list
            if not isinstance(contribution, mol_type)
        ]
        contribution_types = [type(contribution) for contribution in self.contribution_list]
        molecular = None
        if mol_type in contribution_types:
            molecular = self.contribution_list[contribution_types.index(mol_type)]

        _mu = 1.0 / self._mu_quads[:, None]
        _w = self._wi_quads[:, None]

        for contribution in non_molecular:
            contribution.contribute(
                self, 0, total_layers, 0, 0, density, surface_tau, path_length=dz
            )

        surface_tau = surface_tau * _mu
        if molecular is not None:
            for index, mu in enumerate(_mu):
                tmp_tau[...] = 0.0
                molecular.contribute(
                    self, 0, total_layers, 0, 0, density, tmp_tau, path_length=dz * mu
                )
                surface_tau[index] += tmp_tau[0]

        intensity = black_body(wngrid, temperature[0]) / PI * np.exp(-surface_tau)

        for layer in range(total_layers):
            layer_tau[...] = 0.0
            dtau[...] = 0.0
            for contribution in non_molecular:
                contribution.contribute(
                    self,
                    layer + 1,
                    total_layers,
                    0,
                    0,
                    density,
                    layer_tau,
                    path_length=dz,
                )
                contribution.contribute(
                    self, layer, layer + 1, 0, 0, density, dtau, path_length=dz
                )

            k_dtau = None
            k_layer = None
            weights = None
            if molecular is not None:
                weights = molecular.weights
                ngauss = len(weights)
                sigma = molecular.sigma_xsec
                k_layer = contribute_ktau_emission(
                    layer + 1,
                    total_layers,
                    0,
                    sigma,
                    density,
                    dz,
                    weights,
                    wngrid_size,
                    0,
                    ngauss,
                )
                k_dtau = contribute_ktau_emission(
                    layer,
                    layer + 1,
                    0,
                    sigma,
                    density,
                    dz,
                    weights,
                    wngrid_size,
                    0,
                    ngauss,
                )
                k_dtau += k_layer

            dtau += layer_tau
            planck = (
                black_body(wngrid, temperature[layer])
                / PI
                * (layer_radius[layer] ** 2)
                / (planet_radius**2)
            )

            dtau_calc = np.exp(-dtau * _mu)
            layer_tau_calc = np.exp(-layer_tau * _mu)
            if molecular is not None and weights is not None and k_dtau is not None and k_layer is not None:
                dtau_calc *= np.sum(np.exp(-k_dtau * _mu[:, None]) * weights, axis=-1)
                layer_tau_calc *= np.sum(np.exp(-k_layer * _mu[:, None]) * weights, axis=-1)

            intensity += planck * (layer_tau_calc - dtau_calc)

            dtau_calc = np.exp(-dtau) if dtau.min() < self._clamp else 0.0
            layer_tau_calc = np.exp(-layer_tau) if layer_tau.min() < self._clamp else 0.0
            if molecular is not None and weights is not None and k_dtau is not None and k_layer is not None:
                if k_dtau.min() < self._clamp:
                    dtau_calc *= np.sum(np.exp(-k_dtau) * weights, axis=-1)
                if k_layer.min() < self._clamp:
                    layer_tau_calc *= np.sum(np.exp(-k_layer) * weights, axis=-1)

            _tau = layer_tau_calc - dtau_calc
            tau[layer] += _tau if isinstance(_tau, float) else _tau[0]

        return intensity, _mu, _w, tau

    def evaluate_emission(
        self, wngrid: npt.NDArray[np.float64], return_contrib: bool
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        if self.usingKTables:
            return self.evaluate_emission_ktables(wngrid, return_contrib)

        dz = self.deltaz
        planet_radius = self._planet.fullRadius
        layer_radius = self.altitudeProfile + self._planet.fullRadius
        total_layers = self.nLayers
        density = self.densityProfile
        wngrid_size = wngrid.shape[0]
        temperature = self.temperatureProfile
        tau = np.zeros(shape=(self.nLayers, wngrid_size))
        surface_tau = np.zeros(shape=(1, wngrid_size))
        layer_tau = np.zeros(shape=(1, wngrid_size))
        dtau = np.zeros(shape=(1, wngrid_size))

        for contribution in self.contribution_list:
            contribution.contribute(
                self, 0, total_layers, 0, 0, density, surface_tau, path_length=dz
            )

        planck = black_body(wngrid, temperature[0]) / PI
        _mu = 1.0 / self._mu_quads[:, None]
        _w = self._wi_quads[:, None]
        intensity = planck * np.exp(-surface_tau * _mu)

        for layer in range(total_layers):
            layer_tau[...] = 0.0
            dtau[...] = 0.0
            for contribution in self.contribution_list:
                contribution.contribute(
                    self,
                    layer + 1,
                    total_layers,
                    0,
                    0,
                    density,
                    layer_tau,
                    path_length=dz,
                )
                contribution.contribute(
                    self, layer, layer + 1, 0, 0, density, dtau, path_length=dz
                )

            dtau += layer_tau
            dtau_calc = np.exp(-dtau) if dtau.min() < self._clamp else 0.0
            layer_tau_calc = np.exp(-layer_tau) if layer_tau.min() < self._clamp else 0.0
            _tau = layer_tau_calc - dtau_calc
            tau[layer] += _tau if isinstance(_tau, float) else _tau[0]

            planck = (
                black_body(wngrid, temperature[layer])
                / PI
                * (layer_radius[layer] ** 2)
                / (planet_radius**2)
            )
            intensity += planck * (np.exp(-layer_tau * _mu) - np.exp(-dtau * _mu))

        return intensity, _mu, _w, tau

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("emission_radscale", "eclipse_radscale")


class MultiDirectImModel(MultiTransitModel):
    """Weighted combination of multiple direct-imaging submodels."""

    def __init__(self, *args: t.Any, radius_scaling: bool = False, **kwargs: t.Any) -> None:
        self._radius_scaling = radius_scaling
        super().__init__(*args, **kwargs)

    def select_model(self) -> t.Type[ForwardModel]:
        model_class: t.Type[ForwardModel] = DirectImageModel
        if self._use_cuda:
            try:
                from taurex_cuda import DirectImageCudaModel

                model_class = DirectImageCudaModel
            except (ModuleNotFoundError, ImportError):
                self.warning("Cuda plugin not found or not working")
        if self._radius_scaling:
            model_class = DirectImageRadiusScaleModel
        return model_class

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("multidirectimage",)


class DirectImageRadiusScaleModel(EmissionModelRadiusScale):
    """Direct-image variant of the radius-scaled emission model."""

    def compute_final_flux(self, f_total: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return compute_direct_image_final_flux(
            f_total, self._planet.fullRadius, self._star.distance * 3.08567758e16
        )

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("direct_radscale", "directimage_radscale")


class BaseParameterTransitModel(ForwardModel):
    """Composite model wrapper that reads per-region parameter files."""

    def __init__(
        self,
        name: str,
        planet: t.Optional[Planet] = None,
        star: t.Optional[Star] = None,
        observation: t.Optional[BaseSpectrum] = None,
        pressure_profile: t.Optional[PressureProfile] = None,
        temperature_profile: t.Optional[TemperatureProfile] = None,
        chemistry: t.Optional[Chemistry] = None,
        nlayers: int = 100,
        atm_min_pressure: float = 1e-4,
        atm_max_pressure: float = 1e6,
        parfiles: t.Optional[t.Sequence[str]] = None,
        use_cuda: bool = False,
    ) -> None:
        super().__init__(name)
        self._planet = planet
        self._star = star
        self._observation = observation
        self._default_pressure = pressure_profile or SimplePressureProfile(
            nlayers, atm_min_pressure, atm_max_pressure
        )
        self._atm_min_pressure = atm_min_pressure
        self._atm_max_pressure = atm_max_pressure
        self._nlayers = nlayers
        self._default_temperature = temperature_profile
        self._default_chemistry = chemistry
        self._parfiles = list(parfiles or [])
        self._use_cuda = use_cuda
        self._multimodel: t.Optional[MultiTransitModel] = None

    def defaultBinner(self) -> Binner:  # noqa: N802
        if self._multimodel is None:
            return super().defaultBinner()
        return self._multimodel.defaultBinner()

    def read_parameters(
        self, parfile: t.Optional[str]
    ) -> t.Tuple[
        t.Optional[TemperatureProfile],
        t.Optional[Chemistry],
        PressureProfile,
        t.List[Contribution],
    ]:
        from taurex.parameter import ParameterParser
        from taurex.parameter.factory import generate_contributions

        if parfile is None:
            return (
                self._default_temperature,
                self._default_chemistry,
                self._default_pressure,
                list(self.contribution_list),
            )

        parser = ParameterParser()
        parser.read(parfile)

        temperature = parser.generate_temperature_profile() or self._default_temperature
        chemistry = parser.generate_chemistry_profile() or self._default_chemistry
        pressure = parser.generate_pressure_profile() or self._default_pressure
        try:
            contributions = generate_contributions(parser._raw_config["Model"])
        except KeyError:
            contributions = []
        if len(contributions) == 0:
            contributions = list(self.contribution_list)
        return temperature, chemistry, pressure, contributions

    def setup_keywords(self) -> t.Dict[str, t.Any]:
        regions = [self.read_parameters(parfile) for parfile in self._parfiles]
        if not regions:
            regions = [
                (
                    self._default_temperature,
                    self._default_chemistry,
                    self._default_pressure,
                    list(self.contribution_list),
                )
            ]
        temperature_profiles, chemistry, pressure, contributions = zip(*regions)
        nlayers = [region_pressure.nLayers for region_pressure in pressure]
        return {
            "temperature_profiles": list(temperature_profiles),
            "chemistry": list(chemistry),
            "pressure_profile": list(pressure),
            "planet": self._planet,
            "star": self._star,
            "observation": self._observation,
            "contributions": list(contributions),
            "pressure_min": self._atm_min_pressure,
            "pressure_max": self._atm_max_pressure,
            "nlayers": nlayers,
            "use_cuda": self._use_cuda,
        }

    def initialize_profiles(self) -> None:
        if self._multimodel is None:
            raise RuntimeError("Model must be built before initializing profiles")
        self._multimodel.initialize_profiles()

    def create_model(self) -> MultiTransitModel:
        raise NotImplementedError

    def build(self) -> None:
        self._multimodel = self.create_model()
        self._multimodel.build()
        self._fitting_parameters = dict(self._multimodel.fittingParameters)
        self._derived_parameters = dict(self._multimodel.derivedParameters)

    @property
    def densityProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return self._multimodel.densityProfile

    @property
    def altitudeProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return self._multimodel.altitudeProfile

    @property
    def temperatureProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return self._multimodel.temperatureProfile

    @property
    def pressureProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return self._multimodel.pressureProfile

    @property
    def chemistry(self) -> MultiChemistry:
        return self._multimodel.chemistry

    @property
    def scaleheight_profile(self) -> npt.NDArray[np.float64]:
        return self._multimodel.scaleheight_profile

    @property
    def gravity_profile(self) -> npt.NDArray[np.float64]:
        return self._multimodel.gravity_profile

    def model(
        self,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        cutoff_grid: t.Optional[bool] = True,
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        t.Optional[npt.NDArray[np.float64]],
        t.Optional[t.Any],
    ]:
        return self._multimodel.model(wngrid=wngrid, cutoff_grid=cutoff_grid)

    def compute_error(
        self,
        samples: t.Callable[[], t.Iterable[float]],
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        binner: t.Optional[Binner] = None,
    ) -> t.Tuple[t.Dict[str, npt.NDArray[np.float64]], t.Dict[str, npt.NDArray[np.float64]]]:
        return self._multimodel.compute_error(samples, wngrid=wngrid, binner=binner)

    def write(self, output: OutputGroup) -> OutputGroup:
        return self._multimodel.write(output)

    @property
    def nativeWavenumberGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return self._multimodel.nativeWavenumberGrid


class MultiParameterTransitModel(BaseParameterTransitModel):
    """Parameter-file-driven composite transmission model."""

    def __init__(
        self,
        planet: t.Optional[Planet] = None,
        star: t.Optional[Star] = None,
        observation: t.Optional[BaseSpectrum] = None,
        pressure_profile: t.Optional[PressureProfile] = None,
        temperature_profile: t.Optional[TemperatureProfile] = None,
        chemistry: t.Optional[Chemistry] = None,
        nlayers: int = 100,
        atm_min_pressure: float = 1e-4,
        atm_max_pressure: float = 1e6,
        parfiles: t.Optional[t.Sequence[str]] = None,
        use_cuda: bool = False,
        fractions: t.Optional[t.Sequence[float]] = None,
    ) -> None:
        super().__init__(
            "MultiTransitParameter",
            planet=planet,
            star=star,
            observation=observation,
            pressure_profile=pressure_profile,
            temperature_profile=temperature_profile,
            chemistry=chemistry,
            nlayers=nlayers,
            atm_min_pressure=atm_min_pressure,
            atm_max_pressure=atm_max_pressure,
            parfiles=parfiles,
            use_cuda=use_cuda,
        )
        self._fractions = list(fractions or []) if fractions is not None else None

    def create_model(self) -> MultiTransitModel:
        kwargs = self.setup_keywords()
        return MultiTransitModel(**kwargs, fractions=self._fractions)

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("multi_transit",)


class MultiParameterEclipseModel(BaseParameterTransitModel):
    """Parameter-file-driven composite emission model."""

    def __init__(
        self,
        planet: t.Optional[Planet] = None,
        star: t.Optional[Star] = None,
        observation: t.Optional[BaseSpectrum] = None,
        pressure_profile: t.Optional[PressureProfile] = None,
        temperature_profile: t.Optional[TemperatureProfile] = None,
        chemistry: t.Optional[Chemistry] = None,
        nlayers: int = 100,
        atm_min_pressure: float = 1e-4,
        atm_max_pressure: float = 1e6,
        parfiles: t.Optional[t.Sequence[str]] = None,
        use_cuda: bool = False,
        fractions: t.Optional[t.Sequence[float]] = None,
        radius_scaling: bool = False,
    ) -> None:
        super().__init__(
            "MultiEclipseParameter",
            planet=planet,
            star=star,
            observation=observation,
            pressure_profile=pressure_profile,
            temperature_profile=temperature_profile,
            chemistry=chemistry,
            nlayers=nlayers,
            atm_min_pressure=atm_min_pressure,
            atm_max_pressure=atm_max_pressure,
            parfiles=parfiles,
            use_cuda=use_cuda,
        )
        self._fractions = list(fractions or []) if fractions is not None else None
        self._radius_scaling = radius_scaling

    def create_model(self) -> MultiEclipseModel:
        kwargs = self.setup_keywords()
        return MultiEclipseModel(
            **kwargs, fractions=self._fractions, radius_scaling=self._radius_scaling
        )

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("multi_eclipse",)


class MultiParameterDirectImModel(BaseParameterTransitModel):
    """Parameter-file-driven composite direct-image model."""

    def __init__(
        self,
        planet: t.Optional[Planet] = None,
        star: t.Optional[Star] = None,
        observation: t.Optional[BaseSpectrum] = None,
        pressure_profile: t.Optional[PressureProfile] = None,
        temperature_profile: t.Optional[TemperatureProfile] = None,
        chemistry: t.Optional[Chemistry] = None,
        nlayers: int = 100,
        atm_min_pressure: float = 1e-4,
        atm_max_pressure: float = 1e6,
        parfiles: t.Optional[t.Sequence[str]] = None,
        use_cuda: bool = False,
        fractions: t.Optional[t.Sequence[float]] = None,
        radius_scaling: bool = False,
    ) -> None:
        super().__init__(
            "MultiDirectParameter",
            planet=planet,
            star=star,
            observation=observation,
            pressure_profile=pressure_profile,
            temperature_profile=temperature_profile,
            chemistry=chemistry,
            nlayers=nlayers,
            atm_min_pressure=atm_min_pressure,
            atm_max_pressure=atm_max_pressure,
            parfiles=parfiles,
            use_cuda=use_cuda,
        )
        self._fractions = list(fractions or []) if fractions is not None else None
        self._radius_scaling = radius_scaling

    def create_model(self) -> MultiDirectImModel:
        kwargs = self.setup_keywords()
        return MultiDirectImModel(
            **kwargs, fractions=self._fractions, radius_scaling=self._radius_scaling
        )

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("multi_directimage",)