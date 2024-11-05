"""Optimizer using multinest library."""

import os
import pathlib
import typing as t

import numpy as np
import numpy.typing as npt

from taurex import OutputSize
from taurex.model import ForwardModel
from taurex.mpi import barrier, get_rank
from taurex.output import OutputGroup
from taurex.spectrum import BaseSpectrum
from taurex.types import AnyValType, PathLike
from taurex.util import (
    quantile_corner,
    read_error_into_dict,
    read_table,
    recursively_save_dict_contents_to_output,
)

from .optimizer import FitParam, Optimizer

NestMarginalOutput = t.TypedDict(
    "NestMarginalOutput",
    {
        "median": float,
        "sigma": float,
        "1sigma": float,
        "2sigma": float,
        "3sigma": float,
        "5sigma": float,
        "q75%": float,
        "q25%": float,
        "q95%": float,
        "q01%": float,
        "q90%": float,
        "q10%": float,
    },
)
"""Marginal output."""
NestModeStatsOutput = t.TypedDict(
    "NestModeStatsOutput",
    {
        "Strictly Local Log-Evidence": t.Optional[float],
        "Strictly Local Log-Evidence error": t.Optional[float],
        "Local Log-Evidence": t.Optional[float],
        "Local Log-Evidence error": t.Optional[float],
        "mean": t.List[float],
        "sigma": t.List[float],
        "maximum": t.List[float],
        "maximum a posterior": t.List[float],
    },
)
"""Nested Mode stats output"""


NestStatsOutput = t.TypedDict(
    "NestStatsOutput",
    {
        "modes": t.List[NestModeStatsOutput],
        "marginals": t.List[NestMarginalOutput],
        "global evidence": t.Optional[float],
        "global evidence error": t.Optional[float],
    },
)


class NestFitParam(FitParam):
    """Fit parameter for multinest."""

    nest_map: float
    nest_median: float


class NestSolutionOutput(t.TypedDict):
    type: str
    local_logE: t.Tuple[float, float]  # noqa: N815
    weights: npt.NDArray[np.float64]
    tracedata: npt.NDArray[np.float64]
    fit_params: t.Dict[str, NestFitParam]


class NestOutputType(t.TypedDict):
    """Multinest output type"""

    NEST_stats: NestStatsOutput
    global_logE: t.Tuple[float, float]  # noqa: N815
    solutions: t.Dict[str, NestSolutionOutput]


class MultiNestOptimizer(Optimizer):
    def __init__(
        self,
        multi_nest_path: t.Optional[PathLike] = None,
        observed: t.Optional[BaseSpectrum] = None,
        model: t.Optional[ForwardModel] = None,
        sampling_efficiency: t.Optional[t.Literal["parameter"]] = "parameter",
        num_live_points: t.Optional[int] = 1500,
        max_iterations: t.Optional[int] = 0,
        search_multi_modes: t.Optional[bool] = True,
        num_params_cluster: t.Optional[int] = None,
        maximum_modes: t.Optional[int] = 100,
        constant_efficiency_mode: t.Optional[bool] = False,
        evidence_tolerance: t.Optional[float] = 0.5,
        mode_tolerance: t.Optional[float] = -1e90,
        importance_sampling: t.Optional[bool] = False,
        resume: t.Optional[bool] = False,
        only_finish: t.Optional[bool] = False,
        n_iter_before_update: t.Optional[int] = 100,
        multinest_prefix: t.Optional[str] = "1-",
        verbose_output: t.Optional[bool] = True,
        sigma_fraction: t.Optional[float] = 0.1,
    ):
        r"""Setup multinest optimizer.

        Parameters
        ----------
        multi_nest_path:
            Path to multinest output directory, by default None
        observed:
            Observed spectrum to optimize to
        model:
            Forward model to optimize
        sampling_efficiency:
            Sampling efficiency (parameter, ...)
        num_live_points:
            Number of live points
        max_iterations:
            Maximum no. of iterations (0=inf)
        search_multi_modes:
            Search for multiple modes
        num_params_cluster:
            Parameters on which to cluster
            e.g. if nclust_par = 3, it will cluster on the first 3 parameters only.
            if ncluster_par = -1 it clusters on all parameters
        maximum_modes:
            Maximum number of modes
        constant_efficiency_mode:
            Run in constant efficiency mode
        evidence_tolerance:
            Set log likelihood tolerance. :math:`\Delta \log Z <` evidence_tolerance
        mode_tolerance:
            Mode tolerance
        importance_sampling:
            Use importance nested sampling
        resume:
            Resume from previous run
        n_iter_before_update:
            Number of iterations before updating
        multinest_prefix:
            Prefix for multinest output files
        verbose_output:
            Verbose output
        sigma_fraction:
            Fraction of sigma to use for prior. Default is 10%
        """
        super().__init__("Multinest", observed, model, sigma_fraction)

        multi_nest_path = pathlib.Path(multi_nest_path)
        # sampling chains directory
        self.nest_path = "chains/"
        self.nclust_par = -1
        # sampling efficiency (parameter, ...)
        self.sampling_eff = sampling_efficiency
        # number of live points
        self.n_live_points = int(num_live_points)
        # maximum no. of iterations (0=inf)
        self.max_iter = int(max_iterations)
        # search for multiple modes
        self.multimodes = search_multi_modes
        self.n_iter_before_update = int(n_iter_before_update)
        # parameters on which to cluster, e.g. if nclust_par = 3, it will
        # cluster on the first 3 parameters only.
        # if ncluster_par = -1 it clusters on all parameters
        self.nclust_par = num_params_cluster
        # maximum number of modes
        self.max_modes = int(maximum_modes)
        # run in constant efficiency mode
        self.const_eff = constant_efficiency_mode
        self._only_finish = only_finish
        # set log likelihood tolerance. If change is smaller,
        # multinest will have converged
        self.evidence_tolerance = evidence_tolerance
        self.mode_tolerance = mode_tolerance
        # importance nested sampling
        self.imp_sampling = importance_sampling
        if self.imp_sampling:
            self.multimodes = False

        self.multinest_prefix = multinest_prefix

        if get_rank() == 0:
            if not multi_nest_path.exists():
                self.info("Directory %s does not exist, creating", multi_nest_path)
                multi_nest_path.mkdir(parents=True)
        barrier()

        # Convert to standard string path
        self.dir_multinest = multi_nest_path
        self.info("Found directory %s", self.dir_multinest)

        self.resume = resume
        self.verbose = verbose_output

    def compute_fit(self):
        """Compute the fit."""
        import json

        import pymultinest

        def multinest_loglike(cube, ndim, nparams):
            data = np.array([cube[i] for i in range(len(self.fitting_parameters))])
            return self.log_likelihood(data)

        def multinest_uniform_prior(cube, ndim, nparams):
            # prior distributions called by multinest. Implements a uniform prior
            # converting parameters from normalised grid to uniform prior
            # print(type(cube))
            data = np.array([cube[i] for i in range(nparams)])
            prior = self.prior_transform(data)
            for idx, c in enumerate(prior):
                cube[idx] = c

        ndim = len(self.fitting_parameters)
        self.warning(f"Number of dimensions {ndim}")
        self.warning(f"Fitting parameters {self.fitting_parameters}")

        ncluster = self.nclust_par

        if isinstance(ncluster, float):
            ncluster = int(ncluster)

        if ncluster is not None and ncluster <= 0:
            ncluster = None

        if ncluster is None:
            self.nclust_par = ndim  # For writing to output later on

        # Will help for live plotting
        with open(self.dir_multinest / f"{self.multinest_prefix}params.json", "w") as f:
            json.dump(self.fit_latex, f)

        # write param json file

        self.info("Beginning fit......")

        if not self._only_finish:
            pymultinest.run(
                LogLikelihood=multinest_loglike,
                Prior=multinest_uniform_prior,
                n_dims=ndim,
                multimodal=self.multimodes,
                n_clustering_params=ncluster,
                max_modes=self.max_modes,
                n_iter_before_update=self.n_iter_before_update,
                outputfiles_basename=str(self.dir_multinest / self.multinest_prefix),
                const_efficiency_mode=self.const_eff,
                importance_nested_sampling=self.imp_sampling,
                resume=self.resume,
                verbose=self.verbose,
                sampling_efficiency=self.sampling_eff,
                evidence_tolerance=self.evidence_tolerance,
                mode_tolerance=self.mode_tolerance,
                n_live_points=self.n_live_points,
                max_iter=self.max_iter,
            )

        self.info("Fit complete.....")

        self._multinest_output = self.store_nest_solutions()

        self.debug("Multinest output %s", self._multinest_output)

    def write_optimizer(self, output: OutputGroup) -> OutputGroup:
        """Write optimizer to output."""
        opt = super().write_optimizer(output)

        # sampling efficiency (parameter, ...)
        opt.write_scalar("sampling_eff ", self.sampling_eff)
        # number of live points
        opt.write_scalar("num_live_points", self.n_live_points)
        # maximum no. of iterations (0=inf)
        opt.write_scalar("max_iterations", self.max_iter)
        # search for multiple modes
        opt.write_scalar("search_multimodes", self.multimodes)
        # parameters on which to cluster, e.g. if nclust_par = 3,
        # it will cluster on the first 3 parameters only.
        # if ncluster_par = -1 it clusters on all parameters
        opt.write_scalar("nclust_parameter", self.nclust_par)
        # maximum number of modes
        opt.write_scalar("max_modes", self.max_modes)
        # run in constant efficiency mode
        opt.write_scalar("constant_efficiency", self.const_eff)
        # set log likelihood tolerance. If change is smaller,
        # multinest will have converged
        opt.write_scalar("evidence_tolerance", self.evidence_tolerance)
        opt.write_scalar("mode_tolerance", self.mode_tolerance)
        # importance nested sampling
        opt.write_scalar("importance_sampling ", self.imp_sampling)

        return opt

    def write_fit(self, output):
        fit = super().write_fit(output)

        if self._multinest_output:
            recursively_save_dict_contents_to_output(output, self._multinest_output)

        return fit

    # Laziness brought us to this point
    # This function is so big and I cannot be arsed to rewrite
    # this in a nicer way, if some angel does it
    # for me then I will buy them TWO beers.
    # Also I think pymultinest does most of this so why??!
    def store_nest_solutions(self) -> NestOutputType:  # noqa: C901
        """Store the multinest results."""
        import pymultinest

        self.warning("Store the multinest results")
        nest_out = {"solutions": {}}
        data = np.loadtxt(
            self.dir_multinest
            / f"{self.multinest_prefix}.txt"
            # os.path.join(self.dir_multinest, "{}.txt".format(self.multinest_prefix))
        )

        nest_analyser = pymultinest.Analyzer(
            n_params=len(self.fitting_parameters),
            outputfiles_basename=os.path.join(
                self.dir_multinest, self.multinest_prefix
            ),
        )
        nest_stats = nest_analyser.get_stats()
        nest_stats = t.cast(NestStatsOutput, nest_stats)
        nest_out["NEST_stats"] = nest_stats

        if "global evidence" in nest_stats:
            nest_out["global_logE"] = (
                nest_out["NEST_stats"]["global evidence"],
                nest_out["NEST_stats"]["global evidence error"],
            )

        # Bypass if run in multimodes = False. Pymultinest.Analyser does
        # not report means and sigmas in this case
        if len(nest_out["NEST_stats"]["modes"]) == 0:
            with open(
                self.dir_multinest
                / f"{self.multinest_prefix}stats.dat"
                # os.path.join(
                #     self.dir_multinest, "{}stats.dat".format(self.multinest_prefix)
                # )
            ) as file:
                lines = file.readlines()
            stats = {"modes": []}
            read_error_into_dict(lines[0], stats)

            text = "".join(lines[2:])
            # without INS:
            parts = text.split("\n\n")
            mode = {"index": 0}

            modelines = parts[0]
            tab = read_table(modelines, title="Parameter")
            mode["mean"] = tab[:, 1].tolist()
            mode["sigma"] = tab[:, 2].tolist()

            mode["maximum"] = read_table(parts[1], title="Parameter", d=None)[
                :, 1
            ].tolist()
            mode["maximum a posterior"] = read_table(
                parts[2], title="Parameter", d=None
            )[:, 1].tolist()

            if "Nested Sampling Global Log-Evidence".lower() in stats:
                mode["Local Log-Evidence".lower()] = stats[
                    "Nested Sampling Global Log-Evidence".lower()
                ]
                mode["Local Log-Evidence error".lower()] = stats[
                    "Nested Sampling Global Log-Evidence error".lower()
                ]
            else:
                mode["Local Log-Evidence".lower()] = stats[
                    "Nested Importance Sampling Global Log-Evidence".lower()
                ]
                mode["Local Log-Evidence error".lower()] = stats[
                    "Nested Importance Sampling Global Log-Evidence error".lower()
                ]

            mode["Strictly Local Log-Evidence".lower()] = mode[
                "Local Log-Evidence".lower()
            ]
            mode["Strictly Local Log-Evidence error".lower()] = mode[
                "Local Log-Evidence error".lower()
            ]

            nest_out["NEST_stats"]["modes"] = [mode]

        modes = []
        modes_weights = []
        chains = []
        chains_weights = []

        if self.multimodes:
            # separate modes. get individual samples for each mode

            # get parameter values and sample probability (=weight) for each mode
            with open(
                os.path.join(
                    self.dir_multinest / f"{self.multinest_prefix}post_separate.dat",
                )
            ) as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    if idx > 2:  # skip the first two lines
                        if lines[idx - 1] == "\n" and lines[idx - 2] == "\n":
                            modes.append(chains)
                            modes_weights.append(chains_weights)
                            chains = []
                            chains_weights = []
                    chain = [float(x) for x in line.split()[2:]]
                    if len(chain) > 0:
                        chains.append(chain)
                        chains_weights.append(float(line.split()[0]))
                modes.append(chains)
                modes_weights.append(chains_weights)
            modes_array = []
            for mode in modes:
                mode_array = np.zeros((len(mode), len(mode[0])))
                for idx, line in enumerate(mode):
                    mode_array[idx, :] = line
                modes_array.append(mode_array)
        else:
            # not running in multimode. Get chains directly from file 1-.txt
            modes_array = [data[:, 2:]]
            chains_weights = [data[:, 0]]
            modes_weights.append(chains_weights[0])
            modes = [0]

        for nmode in range(len(modes)):
            self.debug(f"Nmode: {nmode}")

            mydict = {
                "type": "nest",
                "local_logE": (
                    nest_out["NEST_stats"]["modes"][0]["local log-evidence"],
                    nest_out["NEST_stats"]["modes"][0]["local log-evidence error"],
                ),
                "weights": np.asarray(modes_weights[nmode]),
                "tracedata": modes_array[nmode],
                "fit_params": {},
            }

            nest_stats
            for idx, param_name in enumerate(self.fit_names):
                trace = modes_array[nmode][:, idx]
                q_16, q_50, q_84 = quantile_corner(
                    trace, [0.16, 0.5, 0.84], weights=np.asarray(modes_weights[nmode])
                )

                mydict["fit_params"][param_name] = {
                    "value": q_50,
                    "sigma_m": q_50 - q_16,
                    "sigma_p": q_84 - q_50,
                    "nest_map": nest_stats["modes"][nmode]["maximum a posterior"][idx],
                    "mean": nest_stats["modes"][nmode]["mean"][idx],
                    "nest_sigma": nest_stats["modes"][nmode]["sigma"][idx],
                    "trace": trace,
                }

            nest_out["solutions"][f"solution{nmode}"] = mydict

        return nest_out

    def generate_solution(
        self, output_size=OutputSize.heavy
    ) -> t.Dict[str, AnyValType]:
        """Generate solution."""
        solution = super().generate_solution(output_size=output_size)

        solution["GlobalStats"] = self._multinest_output["NEST_stats"]
        return solution

    def get_samples(self, solution_idx: int) -> npt.NDArray[np.float64]:
        """Get samples from solution."""
        return self._multinest_output["solutions"][f"solution{solution_idx}"][
            "tracedata"
        ]

    def get_weights(self, solution_idx: int) -> npt.NDArray[np.float64]:
        """Get weights from solution."""
        return self._multinest_output["solutions"][f"solution{solution_idx}"]["weights"]

    def get_solution(
        self,
    ) -> t.Generator[
        t.Tuple[
            int,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            t.Tuple[
                t.Tuple[
                    t.Literal["Statistics"],
                    t.Dict[
                        t.Literal["local log-evidence", "local log-evidence error"],
                        float,
                    ],
                ],
                t.Tuple[t.Literal["fit_params"], t.Dict[str, NestFitParam]],
                t.Tuple[t.Literal["tracedata"], npt.NDArray[np.float64]],
                t.Tuple[t.Literal["weights"], npt.NDArray[np.float64]],
            ],
        ],
        None,
        None,
    ]:
        """Get solution as generator."""
        names = self.fit_names
        opt_values = self.fit_values
        opt_map = self.fit_values
        solutions = [
            (k, v)
            for k, v in self._multinest_output["solutions"].items()
            if "solution" in k
        ]
        for k, v in solutions:
            solution_idx = int(k[8:])
            for p_name, p_value in v["fit_params"].items():
                idx = names.index(p_name)
                opt_map[idx] = p_value["nest_map"]
                opt_values[idx] = p_value["value"]

            yield solution_idx, opt_map, opt_values, [
                (
                    "Statistics",
                    {
                        "local log-evidence": self._multinest_output["NEST_stats"][
                            "modes"
                        ][solution_idx]["local log-evidence"],
                        "local log-evidence error": self._multinest_output[
                            "NEST_stats"
                        ]["modes"][solution_idx]["local log-evidence error"],
                    },
                ),
                ("fit_params", v["fit_params"]),
                ("tracedata", v["tracedata"]),
                ("weights", v["weights"]),
            ]

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for multinest."""
        return (
            "multinest",
            "pymultinest",
        )

    BIBTEX_ENTRIES = [
        r"""
        @article{ refId0,
        author = {{Buchner, J.} and {Georgakakis, A.} and {Nandra, K.} and {Hsu, L.}
        and {Rangel, C.} and {Brightman, M.} and {Merloni, A.} and {Salvato, M.}
        and {Donley, J.} and {Kocevski, D.}},
        title = {X-ray spectral modelling of the AGN obscuring region in the
            CDFS: Bayesian model selection and catalogue},
        DOI= "10.1051/0004-6361/201322971",
        url= "https://doi.org/10.1051/0004-6361/201322971",
        journal = {A\&A},
        year = 2014,
        volume = 564,
        pages = "A125",
        month = "",
        }
        """,
        """
        @ARTICLE{Feroz_multinest,
            author = {{Feroz}, F. and {Hobson}, M.~P. and {Bridges}, M.},
                title = "{MULTINEST: an efficient and robust Bayesian
                inference tool for cosmology and particle physics}",
            journal = {MNRAS},
            keywords = {methods: data analysis, methods: statistical, Astrophysics},
                year = "2009",
                month = "Oct",
            volume = {398},
            number = {4},
                pages = {1601-1614},
                doi = {10.1111/j.1365-2966.2009.14548.x},
        archivePrefix = {arXiv},
            eprint = {0809.3437},
        primaryClass = {astro-ph},
            adsurl = {https://ui.adsabs.harvard.edu/abs/2009MNRAS.398.1601F},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }
        """,
    ]
