"""Precomputed Mie extinction grids for scalable cloud retrievals."""

import typing as t

import h5py
import numpy as np
import numpy.typing as npt
import scipy.stats as stats

from taurex.exceptions import InvalidModelException
from taurex.output import OutputGroup

from .contribution import Contribution

if t.TYPE_CHECKING:
    from taurex.model.model import ForwardModel
else:
    ForwardModel = object


def _as_list(value: t.Any) -> t.List[t.Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _broadcast_param(
    value: t.Any,
    count: int,
    name: str,
    *,
    allow_none: bool = False,
) -> t.Optional[t.List[t.Any]]:
    if value is None:
        if allow_none:
            return None
        raise InvalidPyMieScattGridException(f"{name} must be provided")

    values = _as_list(value)

    if len(values) == 1 and count > 1:
        return values * count

    if len(values) != count:
        raise InvalidPyMieScattGridException(
            f"{name} must have length 1 or match the number of species ({count})"
        )

    return values


def contribute_mie_tau_numpy(
    start_k: int,
    end_k: int,
    sigma: npt.NDArray[np.float64],
    path: npt.NDArray[np.float64],
    ngrid: int,
    layer: int,
    tau: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    for k in range(start_k, end_k):
        _path = path[k]
        for wn in range(ngrid):
            tau[layer, wn] += sigma[k + layer, wn] * _path

    return tau


try:
    import numba

    contribute_mie_tau = numba.jit(contribute_mie_tau_numpy, nopython=True, nogil=True)
except ImportError:
    contribute_mie_tau = contribute_mie_tau_numpy


class InvalidPyMieScattGridException(InvalidModelException):
    """Raised when the precomputed-grid Mie contribution is misconfigured."""


class PyMieScattGridExtinctionContribution(Contribution):
    """Cloud opacity from precomputed extinction-efficiency grids.

    The supplied HDF5 files are expected to contain a ``radius_grid`` dataset in
    microns, a ``wavenumber_grid`` dataset in :math:`cm^{-1}`, and either a
    ``Qext`` or ``Qext_grid`` dataset containing extinction efficiencies.
    """

    def __init__(
        self,
        mie_particle_mean_radius: t.Optional[t.Any] = None,
        mie_particle_logstd_radius: t.Any = (0.001,),
        mie_particle_paramA: t.Any = (1.0,),
        mie_particle_paramB: t.Any = (6.0,),
        mie_particle_paramC: t.Any = (6.0,),
        mie_particle_paramD: t.Any = (1.0,),
        mie_particle_radius_Nsampling: int = 5,
        mie_particle_radius_Dsampling: float = 2,
        mie_particle_radius_distribution: str = "normal",
        mie_species_path: t.Optional[t.Any] = None,
        species: t.Any = ("Mg2SiO4",),
        mie_particle_mix_ratio: t.Any = (1e-10,),
        mie_porosity: t.Optional[t.Any] = None,
        mie_midP: t.Any = (1e3,),
        mie_rangeP: t.Any = (1.0,),
        mie_nMedium: float = 1,
        mie_resolution: int = 100,
        mie_particle_altitude_distrib: str = "exp_decay",
        mie_particle_altitude_decay: t.Any = (-5.0,),
        name: str = "PyMieScattGridExtinction",
    ) -> None:
        super().__init__(name)

        self._species = _as_list(species)
        self._species_count = len(self._species)

        if self._species_count == 0:
            raise InvalidPyMieScattGridException("At least one species must be provided")

        self._mie_species_path = _broadcast_param(
            mie_species_path, self._species_count, "mie_species_path"
        )
        self._mie_particle_mean_radius = _broadcast_param(
            mie_particle_mean_radius or (0.01,),
            self._species_count,
            "mie_particle_mean_radius",
        )
        self._mie_particle_std_radius = _broadcast_param(
            mie_particle_logstd_radius,
            self._species_count,
            "mie_particle_logstd_radius",
        )
        self._mie_particle_paramA = _broadcast_param(
            mie_particle_paramA, self._species_count, "mie_particle_paramA"
        )
        self._mie_particle_paramB = _broadcast_param(
            mie_particle_paramB, self._species_count, "mie_particle_paramB"
        )
        self._mie_particle_paramC = _broadcast_param(
            mie_particle_paramC, self._species_count, "mie_particle_paramC"
        )
        self._mie_particle_paramD = _broadcast_param(
            mie_particle_paramD, self._species_count, "mie_particle_paramD"
        )
        self._mie_particle_mix_ratio = _broadcast_param(
            mie_particle_mix_ratio,
            self._species_count,
            "mie_particle_mix_ratio",
        )
        self._mie_porosity = _broadcast_param(
            mie_porosity,
            self._species_count,
            "mie_porosity",
            allow_none=True,
        )
        self._mie_midP = _broadcast_param(mie_midP, self._species_count, "mie_midP")
        self._mie_rangeP = _broadcast_param(
            mie_rangeP, self._species_count, "mie_rangeP"
        )
        self._particle_alt_decay = _broadcast_param(
            mie_particle_altitude_decay,
            self._species_count,
            "mie_particle_altitude_decay",
        )

        self._mie_particle_radius_distribution = (
            mie_particle_radius_distribution.lower().strip()
        )
        self._particle_alt_distib = mie_particle_altitude_distrib.lower().strip()
        self._mie_nMedium = mie_nMedium
        self._resolution = mie_resolution
        self._Nsampling = int(mie_particle_radius_Nsampling)
        self._Dsampling = mie_particle_radius_Dsampling

        if self._mie_particle_radius_distribution not in {
            "normal",
            "budaj",
            "deirmendjian",
        }:
            raise InvalidPyMieScattGridException(
                "mie_particle_radius_distribution must be one of: "
                "normal, budaj, deirmendjian"
            )

        if self._particle_alt_distib not in {"exp_decay", "linear"}:
            raise InvalidPyMieScattGridException(
                "mie_particle_altitude_distrib must be 'exp_decay' or 'linear'"
            )

        self._radius_grid, self._Qext, self._Qext_wn = self.load_input_files(
            self._mie_species_path
        )
        self.generate_particle_fitting_params()

    @staticmethod
    def _read_qext_dataset(grid_file: h5py.File, path: str) -> npt.NDArray[np.float64]:
        for dataset_name in ("Qext", "Qext_grid"):
            if dataset_name in grid_file:
                return np.asarray(grid_file[dataset_name][()], dtype=np.float64)

        raise InvalidPyMieScattGridException(
            f"{path} must contain a 'Qext' or 'Qext_grid' dataset"
        )

    def load_input_files(
        self, paths: t.Sequence[str]
    ) -> t.Tuple[
        t.List[npt.NDArray[np.float64]],
        t.List[npt.NDArray[np.float64]],
        t.List[npt.NDArray[np.float64]],
    ]:
        radius_grids = []
        qexts = []
        wavenumber_grids = []

        for path in paths:
            with h5py.File(path, "r") as grid_file:
                try:
                    radius_grid = np.asarray(grid_file["radius_grid"][()], dtype=np.float64)
                    wavenumber_grid = np.asarray(
                        grid_file["wavenumber_grid"][()], dtype=np.float64
                    )
                except KeyError as exc:
                    raise InvalidPyMieScattGridException(
                        f"{path} is missing required dataset {exc!s}"
                    ) from exc

                qext_grid = self._read_qext_dataset(grid_file, path)

            radius_grids.append(radius_grid)
            qexts.append(qext_grid)
            wavenumber_grids.append(wavenumber_grid)

        return radius_grids, qexts, wavenumber_grids

    def contribute(
        self,
        model: ForwardModel,
        start_layer: int,
        end_layer: int,
        density_offset: int,
        layer: int,
        density: npt.NDArray[np.float64],
        tau: npt.NDArray[np.float64],
        path_length: t.Optional[npt.NDArray[np.float64]] = None,
    ):
        contribute_mie_tau(
            start_layer,
            end_layer,
            self.sigma_xsec,
            path_length,
            self._ngrid,
            layer,
            tau,
        )

    def generate_particle_fitting_params(self) -> None:
        bounds_rm = [0.01, 10]
        bounds_rstd = [0.01, 0.2]
        bounds_x = [1e0, 1e12]
        bounds_midp = [1e6, 1e0]
        bounds_rangep = [0.0, 3]
        bounds_decayp = [-7, 0]
        bounds_poro = [0, 1]

        param_name = "Rmean_share"
        param_latex = "$Rmean_share$"

        def read_rmean_share(self):
            return np.mean(self._mie_particle_mean_radius)

        def write_rmean_share(self, value):
            self._mie_particle_mean_radius[:] = [value] * len(self._mie_particle_mean_radius)

        self.add_fittable_param(
            param_name,
            param_latex,
            read_rmean_share,
            write_rmean_share,
            "log",
            False,
            bounds_rm,
        )

        if self._mie_particle_radius_distribution != "budaj":
            param_name = "Rlogstd_share"
            param_latex = "$Rlogstd_share$"

            def read_rstd_share(self):
                return np.mean(self._mie_particle_std_radius)

            def write_rstd_share(self, value):
                self._mie_particle_std_radius[:] = [value] * len(self._mie_particle_std_radius)

            self.add_fittable_param(
                param_name,
                param_latex,
                read_rstd_share,
                write_rstd_share,
                "linear",
                False,
                bounds_rstd,
            )

        param_name = "X_share"
        param_latex = "$X_share$"

        def read_x_share(self):
            return np.mean(self._mie_particle_mix_ratio)

        def write_x_share(self, value):
            self._mie_particle_mix_ratio[:] = [value] * len(self._mie_particle_mix_ratio)

        self.add_fittable_param(
            param_name,
            param_latex,
            read_x_share,
            write_x_share,
            "log",
            False,
            bounds_x,
        )

        param_name = "midP_share"
        param_latex = "$midP_share$"

        def read_midp_share(self):
            return np.mean(self._mie_midP)

        def write_midp_share(self, value):
            self._mie_midP[:] = [value] * len(self._mie_midP)

        self.add_fittable_param(
            param_name,
            param_latex,
            read_midp_share,
            write_midp_share,
            "log",
            False,
            bounds_midp,
        )

        param_name = "rangeP_share"
        param_latex = "$rangeP_share$"

        def read_rangep_share(self):
            return np.mean(self._mie_rangeP)

        def write_rangep_share(self, value):
            self._mie_rangeP[:] = [value] * len(self._mie_rangeP)

        self.add_fittable_param(
            param_name,
            param_latex,
            read_rangep_share,
            write_rangep_share,
            "linear",
            False,
            bounds_rangep,
        )

        param_name = "decayP_share"
        param_latex = "$decayP_share$"

        def read_decayp_share(self):
            return np.mean(self._particle_alt_decay)

        def write_decayp_share(self, value):
            self._particle_alt_decay[:] = [value] * len(self._particle_alt_decay)

        self.add_fittable_param(
            param_name,
            param_latex,
            read_decayp_share,
            write_decayp_share,
            "linear",
            False,
            bounds_decayp,
        )

        for idx, val in enumerate(self._species):
            param_name = f"Rmean_{val}"
            param_latex = f"$Rmean_{val}$"

            def read_rmean(self, idx=idx):
                return self._mie_particle_mean_radius[idx]

            def write_rmean(self, value, idx=idx):
                self._mie_particle_mean_radius[idx] = value

            self.add_fittable_param(
                param_name,
                param_latex,
                read_rmean,
                write_rmean,
                "log",
                False,
                bounds_rm,
            )

            if self._mie_particle_radius_distribution != "budaj":
                param_name = f"Rlogstd_{val}"
                param_latex = f"$Rlogstd_{val}$"

                def read_rstd(self, idx=idx):
                    return self._mie_particle_std_radius[idx]

                def write_rstd(self, value, idx=idx):
                    self._mie_particle_std_radius[idx] = value

                self.add_fittable_param(
                    param_name,
                    param_latex,
                    read_rstd,
                    write_rstd,
                    "linear",
                    False,
                    bounds_rstd,
                )

            if self._mie_porosity is not None:
                param_name = f"Porosity_{val}"
                param_latex = f"$Porosity_{val}$"

                def read_poro(self, idx=idx):
                    return self._mie_porosity[idx]

                def write_poro(self, value, idx=idx):
                    self._mie_porosity[idx] = value

                self.add_fittable_param(
                    param_name,
                    param_latex,
                    read_poro,
                    write_poro,
                    "linear",
                    False,
                    bounds_poro,
                )

            param_name = f"X_{val}"
            param_latex = f"$X_{val}$"

            def read_x(self, idx=idx):
                return self._mie_particle_mix_ratio[idx]

            def write_x(self, value, idx=idx):
                self._mie_particle_mix_ratio[idx] = value

            self.add_fittable_param(
                param_name,
                param_latex,
                read_x,
                write_x,
                "log",
                False,
                bounds_x,
            )

            param_name = f"midP_{val}"
            param_latex = f"$midP_{val}$"

            def read_midp(self, idx=idx):
                return self._mie_midP[idx]

            def write_midp(self, value, idx=idx):
                self._mie_midP[idx] = value

            self.add_fittable_param(
                param_name,
                param_latex,
                read_midp,
                write_midp,
                "log",
                False,
                bounds_midp,
            )

            param_name = f"rangeP_{val}"
            param_latex = f"$rangeP_{val}$"

            def read_rangep(self, idx=idx):
                return self._mie_rangeP[idx]

            def write_rangep(self, value, idx=idx):
                self._mie_rangeP[idx] = value

            self.add_fittable_param(
                param_name,
                param_latex,
                read_rangep,
                write_rangep,
                "linear",
                False,
                bounds_rangep,
            )

            param_name = f"decayP_{val}"
            param_latex = f"$decayP_{val}$"

            def read_decayp(self, idx=idx):
                return self._particle_alt_decay[idx]

            def write_decayp(self, value, idx=idx):
                self._particle_alt_decay[idx] = value

            self.add_fittable_param(
                param_name,
                param_latex,
                read_decayp,
                write_decayp,
                "linear",
                False,
                bounds_decayp,
            )

    def prepare_each(
        self, model: ForwardModel, wngrid: npt.NDArray[np.float64]
    ) -> t.Generator[t.Tuple[str, npt.NDArray[np.float64]], None, None]:
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]
        pressure_profile = model.pressureProfile
        sigma_xsec = np.zeros(shape=(self._nlayers, self._ngrid))

        for specie_idx, _ in enumerate(self._species):
            wn = self._Qext_wn[specie_idx]
            mean_radius = self._mie_particle_mean_radius[specie_idx]

            if self._mie_particle_radius_distribution == "budaj":
                log_rsigma = 0.2
                radii_log = np.linspace(
                    10 ** (np.log10(mean_radius) + self._Dsampling * log_rsigma),
                    10 ** (np.log10(mean_radius) - self._Dsampling * log_rsigma),
                    self._Nsampling,
                )
                weights = ((radii_log / mean_radius) ** 6) * np.exp(
                    -6 * radii_log / mean_radius
                )
            elif self._mie_particle_radius_distribution == "deirmendjian":
                log_rsigma = self._mie_particle_std_radius[specie_idx]
                radii_log = np.linspace(
                    10 ** (np.log10(mean_radius) + self._Dsampling * log_rsigma),
                    10 ** (np.log10(mean_radius) - self._Dsampling * log_rsigma),
                    self._Nsampling,
                )
                weights = self._mie_particle_paramA[specie_idx] * (
                    radii_log ** self._mie_particle_paramB[specie_idx]
                ) * np.exp(
                    -self._mie_particle_paramC[specie_idx]
                    * (radii_log ** self._mie_particle_paramD[specie_idx])
                )
            else:
                log_rsigma = self._mie_particle_std_radius[specie_idx]
                radii_log = np.linspace(
                    10 ** (np.log10(mean_radius) + self._Dsampling * log_rsigma),
                    10 ** (np.log10(mean_radius) - self._Dsampling * log_rsigma),
                    self._Nsampling,
                )
                weights = stats.norm.pdf(
                    np.log10(radii_log), np.log10(mean_radius), log_rsigma
                )

            qexts = []
            radius_grid = self._radius_grid[specie_idx]
            qext_grid = self._Qext[specie_idx]

            for radius in radii_log:
                grid_idx = np.searchsorted(radius_grid, radius) - 1
                grid_idx = int(np.clip(grid_idx, 0, radius_grid.shape[0] - 2))

                radius_1 = radius_grid[grid_idx]
                radius_2 = radius_grid[grid_idx + 1]
                delta_radius = radius_1 - radius_2

                qext_1 = qext_grid[grid_idx]
                qext_2 = qext_grid[grid_idx + 1]

                slope = (qext_1 - qext_2) / delta_radius
                intercept = qext_1 - slope * radius_1
                qexts.append(slope * radius + intercept)

            qexts = np.asarray(qexts) * np.power(radii_log[:, None] * 1e3, 2)
            qext_mean = np.average(qexts, axis=0, weights=weights)
            wn_order = np.argsort(wn)
            qext_interp = np.interp(
                wngrid,
                wn[wn_order],
                qext_mean[wn_order],
                left=0,
                right=0,
            )

            sigma_mie = np.zeros(self._ngrid)
            valid_qext = qext_interp != 0
            sigma_mie[valid_qext] = qext_interp[valid_qext] * np.pi * 1e-18

            if self._mie_midP[specie_idx] == -1:
                bottom_pressure = pressure_profile[0]
                top_pressure = pressure_profile[-1]
            else:
                bottom_pressure = 10 ** (
                    np.log10(self._mie_midP[specie_idx])
                    + self._mie_rangeP[specie_idx] / 2
                )
                top_pressure = 10 ** (
                    np.log10(self._mie_midP[specie_idx])
                    - self._mie_rangeP[specie_idx] / 2
                )

            cloud_filter = (pressure_profile <= bottom_pressure) & (
                pressure_profile >= top_pressure
            )
            sigma_xsec_int = np.zeros((self._nlayers, self._ngrid))

            if self._particle_alt_distib == "exp_decay":
                decay = self._particle_alt_decay[specie_idx]
                mix = self._mie_particle_mix_ratio[specie_idx] * (
                    pressure_profile / bottom_pressure
                ) ** (-decay)
                sigma_xsec_int[cloud_filter, :] = sigma_mie[None, :] * mix[
                    cloud_filter, None
                ]
            else:
                sigma_xsec_int[cloud_filter, :] = (
                    sigma_mie[None, :] * self._mie_particle_mix_ratio[specie_idx]
                )

            sigma_xsec += sigma_xsec_int

        self.sigma_xsec = sigma_xsec
        yield "PyMieScattGridExt", sigma_xsec

    def write(self, output: OutputGroup) -> OutputGroup:
        contrib = super().write(output)
        contrib.write_array("particle_mean_radius", np.array(self._mie_particle_mean_radius))
        if self._mie_particle_radius_distribution != "budaj":
            contrib.write_array(
                "particle_std_radius", np.array(self._mie_particle_std_radius)
            )
        contrib.write_array("particle_mix_ratio", np.array(self._mie_particle_mix_ratio))
        contrib.write_array("particle_midP", np.array(self._mie_midP))
        contrib.write_array("particle_rangeP", np.array(self._mie_rangeP))
        contrib.write_string_array("cloud_species", self._species)
        contrib.write_scalar("radius_Nsampling", self._Nsampling)
        contrib.write_scalar("radius_Dsampling", self._Dsampling)
        contrib.write_scalar("mie_nMedium", self._mie_nMedium)
        return contrib

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("PyMieScattGridExtinction",)

    BIBTEX_ENTRIES = [
        """
        @BOOK{1983asls.book.....B,
               author = {{Bohren}, Craig F. and {Huffman}, Donald R.},
                title = "{Absorption and scattering of light by small particles}",
                 year = 1983,
               adsurl = {https://ui.adsabs.harvard.edu/abs/1983asls.book.....B},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }
        @ARTICLE{2026A&A...707A.127V,
               author = {{Voyer}, M. and {Changeat}, Q.},
                title = "{Precomputed aerosol extinction, scattering, and asymmetry grids for scalable atmospheric retrievals}",
              journal = {Astronomy and Astrophysics},
             keywords = {radiative transfer, methods: numerical, planets and satellites: atmospheres, planets and satellites: gaseous planets, Earth and Planetary Astrophysics, Instrumentation and Methods for Astrophysics},
                 year = 2026,
                month = mar,
               volume = {707},
                  eid = {A127},
                pages = {A127},
                  doi = {10.1051/0004-6361/202558469},
        archivePrefix = {arXiv},
               eprint = {2601.14177},
         primaryClass = {astro-ph.EP},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2026A&A...707A.127V},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }
        """,
    ]