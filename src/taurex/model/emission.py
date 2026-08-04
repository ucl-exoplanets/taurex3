"""Radiative transfer modeeling of eclipses."""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex.chemistry import Chemistry
from taurex.constants import PI
from taurex.core import derivedparam
from taurex.log import setup_log
from taurex.output import OutputGroup
from taurex.planet import Planet
from taurex.pressure import PressureProfile
from taurex.stellar import Star
from taurex.temperature import TemperatureProfile
from taurex.types import get_float_dtype
from taurex.util.emission import black_body

from .simplemodel import OneDForwardModel


if t.TYPE_CHECKING:
    from taurex.contributions import Contribution
else:
    Contribution = object

_log = setup_log(__name__)


def contribute_ktau_emission_numpy(
    start_k: int,
    end_k: int,
    density_offset: int,
    sigma: npt.NDArray[np.float64],
    density: npt.NDArray[np.float64],
    path: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    ngrid: int,
    layer: int,
    ngauss: int,
) -> npt.NDArray[np.float64]:
    """Contribute ktau emission with numpy."""
    _path = path[start_k:end_k]
    _density = density[start_k + density_offset : end_k + density_offset]

    _sigma = sigma[start_k + layer : end_k + layer, :, :]

    # einsum avoids the (k, ngrid, ngauss) intermediate from broadcasting
    return np.einsum("kig,k,k->i", _sigma, _path, _density)


def contribute_ktau_emission_numba(
    start_k: int,
    end_k: int,
    density_offset: int,
    sigma: npt.NDArray[np.float64],
    density: npt.NDArray[np.float64],
    path: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    ngrid: int,
    layer: int,
    ngauss: int,
) -> npt.NDArray[np.float64]:
    """Contribute ktau emission with numba."""
    tau_temp = np.zeros(shape=(ngrid, ngauss))

    for k in range(start_k, end_k):
        _path = path[k]
        _density = density[k + density_offset]
        # for mol in range(nmols):
        for wn in range(ngrid):
            for g in range(ngauss):
                tau_temp[wn, g] += sigma[k + layer, wn, g] * _path * _density
    return tau_temp


try:
    import numba

    contribute_ktau_emission = numba.jit(
        contribute_ktau_emission_numba,
        nopython=True,
        nogil=True,
        fastmath=True,
    )
except ImportError:
    _log.warning("Numba not installed, using numpy")
    contribute_ktau_emission = contribute_ktau_emission_numpy


class EmissionModel(OneDForwardModel):
    """A forward model for eclipses."""

    def __init__(
        self,
        planet: t.Optional[Planet] = None,
        star: t.Optional[Star] = None,
        pressure_profile: t.Optional[PressureProfile] = None,
        temperature_profile: t.Optional[TemperatureProfile] = None,
        chemistry: t.Optional[Chemistry] = None,
        nlayers: t.Optional[int] = 100,
        atm_min_pressure: t.Optional[float] = 1e-4,
        atm_max_pressure: t.Optional[float] = 1e6,
        contributions: t.Optional[t.List[Contribution]] = None,
        ngauss: t.Optional[int] = 4,
    ):
        """Initialise eclipse forward model.

        This is the classic emission model using quadrature integration.

        Parameters
        ----------
        planet:
            Planet model, default planet is Jupiter

        star:
            Star model, default star is Sun-like

        pressure_profile:
            Pressure model, alternative is to set ``nlayers``,
            ``atm_min_pressure`` and ``atm_max_pressure``

        temperature_profile:
            Temperature model, default is an
            :class:`~taurex.data.profiles.temperature.isothermal.Isothermal`
            profile at 1500 K

        chemistry:
            Chemistry model, default is
            :class:`~taurex.data.profiles.chemistry.taurexchemistry.TaurexChemistry`
            with ``H2O`` and ``CH4``

        nlayers:
            Number of layers. Used if ``pressure_profile`` is not defined.

        atm_min_pressure:
            Pressure at TOA. Used if ``pressure_profile`` is not defined.

        atm_max_pressure:
            Pressure at BOA. Used if ``pressure_profile`` is not defined.

        contributions:
            List of contribution objects to include in the model.

        ngauss:
            Number of gaussian quadrature points, default = 4

        """
        super().__init__(
            self.__class__.__name__,
            planet,
            star,
            pressure_profile,
            temperature_profile,
            chemistry,
            nlayers,
            atm_min_pressure,
            atm_max_pressure,
            contributions,
        )

        self.set_num_gauss(ngauss)
        self._clamp = 10
        # Cached reusable arrays to avoid repeated allocation
        self._cached_tau = None
        self._cached_surface_tau = None
        self._cached_layer_tau = None
        self._cached_dtau = None
        self._cached_tmp_tau = None
        self._cached_wngrid_size = -1
        self._cached_ngauss = -1
        # Buffers for in-place np.exp() calls in hot layer loop
        self._cached_exp_mu = None  # (ngauss, wngrid) for exp(-dtau * _mu)
        self._cached_exp_mu2 = None  # (ngauss, wngrid) for exp(-layer_tau * _mu)

    def _ensure_cached_arrays(self, nlayers, wngrid_size):
        """Allocate or reuse cached work arrays.

        Reuses existing arrays when wngrid_size hasn't changed, avoiding
        repeated large allocations in parallel MPI contexts.
        """
        if self._cached_wngrid_size != wngrid_size:
            dtype = get_float_dtype()
            self._cached_tau = np.empty((nlayers, wngrid_size), dtype=dtype)
            self._cached_surface_tau = np.empty((1, wngrid_size), dtype=dtype)
            self._cached_layer_tau = np.empty((1, wngrid_size), dtype=dtype)
            self._cached_dtau = np.empty((1, wngrid_size), dtype=dtype)
            self._cached_tmp_tau = np.empty((1, wngrid_size), dtype=dtype)
            self._cached_wngrid_size = wngrid_size
        if self._cached_ngauss != self._ngauss:
            dtype = get_float_dtype()
            self._cached_exp_mu = np.empty((self._ngauss, wngrid_size), dtype=dtype)
            self._cached_exp_mu2 = np.empty((self._ngauss, wngrid_size), dtype=dtype)
            self._cached_ngauss = self._ngauss

    def set_num_gauss(
        self,
        value: int,
        coeffs: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Set number of gaussian quadrature points.

        Parameters
        ----------
        value: int
            Number of gaussian quadrature points

        coeffs: np.ndarray, optional
            Coefficients for each gaussian quadrature point

        """
        self._ngauss = int(value)

        mu, weight = np.polynomial.legendre.leggauss(self._ngauss)
        self.set_quadratures(mu, weight, coeffs)

    def set_quadratures(
        self,
        mu: npt.NDArray[np.float64],
        weight: npt.NDArray[np.float64],
        coeffs: t.Optional[npt.NDArray[np.float64]] = None,
    ):
        """Set quadrature points.

        Parameters
        ----------
        mu: np.ndarray
            Quadrature points

        weight: np.ndarray
            Quadrature weights

        coeffs: np.ndarray, optional
            Coefficients for each gaussian quadrature point

        """
        self._mu_quads = (mu + 1) / 2
        self._wi_quads = weight / 2
        self._coeffs = coeffs
        if coeffs is None:
            self._coeffs = np.ones(self._ngauss, dtype=get_float_dtype())

    def compute_final_flux(
        self, f_total: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Compute final flux value.

        For emission this converts the flux emitted to:

        .. math::

            F = \frac{F_{total}}{F_{star}}
            \left(\frac{R_{planet}}{R_{star}}\right)^2

        Parameters
        ----------
        f_total: np.ndarray
            Flux emitted by the planet



        """
        star_sed = self._star.spectralEmissionDensity

        self.debug("Star SED: %s", star_sed)
        star_radius = self._star.radius
        planet_radius = self._planet.fullRadius
        self.debug("star_radius %s", self._star.radius)
        self.debug("planet_radius %s", self._star.radius)
        last_flux = ((f_total + self.albedoterm) / star_sed) * (
            planet_radius / star_radius
        ) ** 2

        self.debug("last_flux %s", last_flux)

        return last_flux

    @property
    def albedoterm(self) -> float:
        """Albedo term."""
        return 0.0

    def partial_model(
        self,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        cutoff_grid: t.Optional[bool] = True,
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Evaluate the flux and quandratues but do not compute the final flux.

        Parameters
        ----------
        wngrid : npt.NDArray[np.float64], optional
            Wavenumber grid, by default None
        cutoff_grid : bool, optional
            Whether to cutoff the grid, by default True

        Returns
        -------
        t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
        npt.NDArray[np.float64], npt.NDArray[np.float64]]
            Intensity, mu, weights, and tau.

        """
        from taurex.util import clip_native_to_wngrid

        self.initialize_profiles()

        native_grid = self.nativeWavenumberGrid
        if wngrid is not None and cutoff_grid:
            native_grid = clip_native_to_wngrid(native_grid, wngrid)
        self._star.initialize(native_grid)

        return self.evaluate_emission(native_grid, False)

    def evaluate_emission_ktables(  # noqa: C901
        self, wngrid: npt.NDArray[np.float64], return_contrib: bool
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Evaluate emission flux on quadratures using ktables.

        Parameters
        ----------
        wngrid : npt.NDArray[np.float64]
            Wavenumber grid.
        return_contrib : bool
            Whether to return contributions.

        Returns
        -------
        t.Tuple
            Intensity, mu, weights, and tau.

        """
        from taurex.contributions import AbsorptionContribution

        dz = self.deltaz
        total_layers = self.nLayers
        density = self.densityProfile
        wngrid_size = wngrid.shape[0]
        temperature = self.temperatureProfile

        # Reuse cached arrays to avoid repeated allocation in parallel runs
        self._ensure_cached_arrays(total_layers, wngrid_size)
        tau = self._cached_tau
        tau.fill(0.0)
        surface_tau = self._cached_surface_tau
        surface_tau.fill(0.0)
        layer_tau = self._cached_layer_tau
        dtau = self._cached_dtau
        tmp_tau = self._cached_tmp_tau

        mol_type = AbsorptionContribution

        # Prepare contributions on-demand (memory-efficient)
        for contrib in self.contribution_list:
            if contrib.sigma_xsec is None:
                contrib.prepare(self, wngrid)

        non_molecule_absorption = [
            c for c in self.contribution_list if not isinstance(c, mol_type)
        ]
        contrib_types = [type(c) for c in self.contribution_list]
        molecule_absorption = None
        if mol_type in contrib_types:
            molecule_absorption = self.contribution_list[contrib_types.index(mol_type)]

        _mu = 1.0 / self._mu_quads[:, None]
        _w = self._wi_quads[:, None]

        # Do surface first
        # for layer in range(total_layers):
        for contrib in non_molecule_absorption:
            contrib.contribute(
                self,
                0,
                total_layers,
                0,
                0,
                density,
                surface_tau,
                path_length=dz,
            )

        surface_tau = surface_tau * _mu
        if molecule_absorption is not None:
            for idx, m in enumerate(_mu):
                tmp_tau[...] = 0.0
                molecule_absorption.contribute(
                    self,
                    0,
                    total_layers,
                    0,
                    0,
                    density,
                    tmp_tau,
                    path_length=dz * m,
                )
                surface_tau[idx] += tmp_tau[0]

        self.debug("density = %s", density[0])
        self.debug("surface_tau = %s", surface_tau)

        planck_term = black_body(wngrid, temperature[0]) / PI

        intensity = planck_term * (np.exp(-surface_tau))

        for layer in range(total_layers):
            layer_tau[...] = 0.0
            dtau[...] = 0.0
            for contrib in non_molecule_absorption:
                contrib.contribute(
                    self,
                    layer + 1,
                    total_layers,
                    0,
                    0,
                    density,
                    layer_tau,
                    path_length=dz,
                )
                contrib.contribute(
                    self,
                    layer,
                    layer + 1,
                    0,
                    0,
                    density,
                    dtau,
                    path_length=dz,
                )

            k_dtau = None
            k_layer = None
            wg = None
            if molecule_absorption is not None:
                wg = molecule_absorption.weights
                ng = len(wg)
                sigma = molecule_absorption.sigma_xsec

                k_layer = contribute_ktau_emission(
                    layer + 1,
                    total_layers,
                    0,
                    sigma,
                    density,
                    dz,
                    wg,
                    wngrid_size,
                    0,
                    ng,
                )

                k_dtau = contribute_ktau_emission(
                    layer,
                    layer + 1,
                    0,
                    sigma,
                    density,
                    dz,
                    wg,
                    wngrid_size,
                    0,
                    ng,
                )
                k_dtau += k_layer

            dtau += layer_tau

            self.debug("dtau[%s]=%s", layer, dtau)
            planck_term = black_body(wngrid, temperature[layer]) / PI
            self.debug(
                "planck_term[%s]=%s,%s",
                layer,
                temperature[layer],
                planck_term,
            )

            dtau_calc = np.exp(-dtau * _mu, out=self._cached_exp_mu)
            layer_tau_calc = np.exp(-layer_tau * _mu, out=self._cached_exp_mu2)

            if molecule_absorption is not None:
                dtau_calc *= np.sum(np.exp(-k_dtau * _mu[:, None]) * wg, axis=-1)
                layer_tau_calc *= np.sum(np.exp(-k_layer * _mu[:, None]) * wg, axis=-1)

            intensity += planck_term * (layer_tau_calc - dtau_calc)

            dtau_calc = 0.0
            if dtau.min() < self._clamp:
                # In-place: reuse surface_tau buffer (idle during layer loop)
                np.exp(-dtau, out=self._cached_surface_tau)
                dtau_calc = self._cached_surface_tau
            layer_tau_calc = 0.0
            if layer_tau.min() < self._clamp:
                # In-place: reuse tmp_tau buffer (idle during layer loop)
                np.exp(-layer_tau, out=self._cached_tmp_tau)
                layer_tau_calc = self._cached_tmp_tau

            if molecule_absorption is not None:
                if k_dtau.min() < self._clamp:
                    dtau_calc *= np.sum(np.exp(-k_dtau) * wg, axis=-1)
                if k_layer.min() < self._clamp:
                    layer_tau_calc *= np.sum(np.exp(-k_layer) * wg, axis=-1)

            _tau = layer_tau_calc - dtau_calc

            if isinstance(_tau, float):
                tau[layer] += _tau
            else:
                tau[layer] += _tau[0]

        self.debug("intensity: %s", intensity)

        # Clean up large sigma_xsec arrays
        for contrib in self.contribution_list:
            if contrib.sigma_xsec is not None:
                del contrib.sigma_xsec
                contrib.sigma_xsec = None

        return intensity, _mu, _w, tau

    @property
    def usingKTables(self) -> bool:  # noqa: N802
        """Whether ktables are being used."""
        from taurex.cache import GlobalCache

        return GlobalCache()["opacity_method"] == "ktables"

    def evaluate_emission(  # noqa: C901
        self, wngrid: npt.NDArray[np.float64], return_contrib: bool
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Evaluate emission flux on quadratures.

        Parameters
        ----------
        wngrid : npt.NDArray[np.float64]
            Wavenumber grid.
        return_contrib : bool
            Whether to return contributions.

        Returns
        -------
        t.Tuple
            Intensity, mu, weights, and tau.

        """
        if self.usingKTables:
            return self.evaluate_emission_ktables(wngrid, return_contrib)

        # Prepare contributions on-demand (memory-efficient)
        for contrib in self.contribution_list:
            if contrib.sigma_xsec is None:
                contrib.prepare(self, wngrid)

        dz = self.deltaz

        total_layers = self.nLayers

        density = self.densityProfile

        wngrid_size = wngrid.shape[0]

        temperature = self.temperatureProfile

        # Reuse cached arrays to avoid repeated allocation in parallel runs
        self._ensure_cached_arrays(total_layers, wngrid_size)
        tau = self._cached_tau
        tau.fill(0.0)
        surface_tau = self._cached_surface_tau
        surface_tau.fill(0.0)
        layer_tau = self._cached_layer_tau
        dtau = self._cached_dtau

        # Do surface first
        # for layer in range(total_layers):
        for contrib in self.contribution_list:
            contrib.contribute(
                self,
                0,
                total_layers,
                0,
                0,
                density,
                surface_tau,
                path_length=dz,
            )
        self.debug("density = %s", density[0])
        self.debug("surface_tau = %s", surface_tau)

        planck_term = black_body(wngrid, temperature[0]) / PI

        _mu = 1.0 / self._mu_quads[:, None]
        _w = self._wi_quads[:, None]
        intensity = planck_term * (np.exp(-surface_tau * _mu))

        self.debug("I1_pre %s", intensity)
        # Loop upwards
        for layer in range(total_layers):
            layer_tau[...] = 0.0
            dtau[...] = 0.0
            for contrib in self.contribution_list:
                contrib.contribute(
                    self,
                    layer + 1,
                    total_layers,
                    0,
                    0,
                    density,
                    layer_tau,
                    path_length=dz,
                )
                contrib.contribute(
                    self,
                    layer,
                    layer + 1,
                    0,
                    0,
                    density,
                    dtau,
                    path_length=dz,
                )
            # for contrib in self.contribution_list:

            self.debug("Layer_tau[%s]=%s", layer, layer_tau)

            dtau += layer_tau

            dtau_calc = 0.0
            if dtau.min() < self._clamp:
                # In-place: reuse surface_tau buffer (idle during layer loop)
                np.exp(-dtau, out=self._cached_surface_tau)
                dtau_calc = self._cached_surface_tau
            layer_tau_calc = 0.0
            if layer_tau.min() < self._clamp:
                # In-place: reuse tmp_tau buffer (idle during layer loop)
                np.exp(-layer_tau, out=self._cached_tmp_tau)
                layer_tau_calc = self._cached_tmp_tau

            _tau = layer_tau_calc - dtau_calc

            if isinstance(_tau, float):
                tau[layer] += _tau
            else:
                tau[layer] += _tau[0]

            self.debug("dtau[%s]=%s", layer, dtau)
            planck_term = black_body(wngrid, temperature[layer]) / PI
            self.debug(
                "planck_term[%s]=%s,%s",
                layer,
                temperature[layer],
                planck_term,
            )

            dtau_calc = 0.0
            if dtau.min() < self._clamp:
                np.exp(-dtau * _mu, out=self._cached_exp_mu)
                dtau_calc = self._cached_exp_mu
            layer_tau_calc = 0.0
            if layer_tau.min() < self._clamp:
                np.exp(-layer_tau * _mu, out=self._cached_exp_mu2)
                layer_tau_calc = self._cached_exp_mu2

            intensity += planck_term * (layer_tau_calc - dtau_calc)

        self.debug("intensity: %s", intensity)

        # Clean up large sigma_xsec arrays
        for contrib in self.contribution_list:
            if contrib.sigma_xsec is not None:
                del contrib.sigma_xsec
                contrib.sigma_xsec = None

        return intensity, _mu, _w, tau

    def path_integral(
        self, wngrid: npt.NDArray[np.float64], return_contrib: bool
    ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Evaluate emission flux and integrate quadratures for flux.

        Parameters
        ----------
        wngrid : npt.NDArray[np.float64]
            Wavenumber grid.
        return_contrib : bool
            Whether to return contributions.

        Returns
        -------
        t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            Flux and tau.

        """
        intensity, _mu, _w, tau = self.evaluate_emission(wngrid, return_contrib)
        self.debug("intensity: %s", intensity)

        flux_total = 2.0 * np.pi * sum(intensity * (_w / _mu))
        self.debug("flux_total %s", flux_total)

        return self.compute_final_flux(flux_total).ravel(), tau

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write model to output group.

        Parameters
        ----------
        output : :class:`~taurex.output.output.OutputGroup`
            Output group to write to.

        Returns
        -------
        :class:`~taurex.output.output.OutputGroup`

        """
        model = super().write(output)
        model.write_scalar("ngauss", self._ngauss)
        return model

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for this class."""
        return (
            "emission",
            "eclipse",
        )

    @derivedparam(
        param_name="log_F_bol",
        param_latex=r"log(F$_\{bol\}$)",
        compute=False,
    )
    def logBolometricFlux(self) -> float:  # noqa: N802
        """log10 Flux integrated over all wavelengths (W m-2)."""
        import math

        try:
            from scipy.integrate import simps
        except ImportError:
            # Using a newer version of scipy where simps was renamed
            from scipy.integrate import simpson as simps

        intensity, _mu, _w, tau = self.partial_model()

        flux_total = 2.0 * np.pi * sum(intensity * (_w / _mu))

        flux_wl = flux_total[::-1]
        wlgrid = 10000 / self.nativeWavenumberGrid[::-1]

        return math.log10(simps(flux_wl, wlgrid))
