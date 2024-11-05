"""Retrieval using nestle library."""
import time
import typing as t

import nestle
import numpy as np
import numpy.typing as npt

from taurex.model import ForwardModel
from taurex.output import OutputGroup
from taurex.spectrum import BaseSpectrum
from taurex.util import quantile_corner, recursively_save_dict_contents_to_output

from .optimizer import FitParamOutput, Optimizer


class NestleStatsOutput(t.TypedDict):
    """Dictionary for storing nestle stats output."""

    LogEvidence: float
    LogEvidenceError: float
    Peakiness: float


class NestleSolutionOutput(t.TypedDict):
    """Dictionary for storing nestle solution output."""

    samples: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]
    fitparams: t.Dict[str, FitParamOutput]


class NestleOptimizer(Optimizer):
    """An optimizer that uses the `nestle <http://kylebarbary.com/nestle/>`_ library
    to perform optimization.

    """

    def __init__(
        self,
        observed: t.Optional[BaseSpectrum] = None,
        model: t.Optional[ForwardModel] = None,
        num_live_points: t.Optional[int] = 1500,
        method: t.Optional[t.Literal["single", "multi", "mcmc"]] = "multi",
        tol: t.Optional[float] = 0.5,
        sigma_fraction: t.Optional[float] = 0.1,
    ):
        """Initialize and setup nestle.


        Parameters
        ----------
        observed:
            Observed spectrum to optimize to

        model:
            Forward model to optimize

        num_live_points:
            Number of live points to use in sampling

        method:
            Nested sampling method to use. ``classic`` uses MCMC exploration,
            ``single`` uses a single ellipsoid and ``multi`` uses
            multiple ellipsoids (similar to Multinest)

        tol:
            Evidence tolerance value to stop the fit. This is based on
            an estimate of the remaining prior volumes
            contribution to the evidence.

        sigma_fraction:
            Fraction of weights to use in computing the error. (Default: 0.1)

        """
        super().__init__("Nestle", observed, model, sigma_fraction)
        self._nlive = int(num_live_points)  # number of live points
        self._method = method  # use MutliNest algorithm

        self._tol = tol  # the stopping criterion
        self._nestle_output: NestleSolutionOutput = None

    @property
    def tolerance(self) -> float:
        """Tolerance value for stopping the fit."""
        return self._tol

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        """Set the tolerance value for stopping the fit."""
        self._tol = value

    @property
    def numLivePoints(self) -> int:  # noqa: N802
        """Number of live points to use in the fit."""
        return self._nlive

    @numLivePoints.setter
    def numLivePoints(self, value: int) -> None:  # noqa: N802
        """Set the number of live points to use in the fit."""
        self._nlive = value

    def compute_fit(self) -> None:
        """Computes the fit using nestle."""

        def nestle_uniform_prior(theta):
            # prior distributions called by multinest. Implements a uniform prior
            # converting parameters from normalised grid to uniform prior

            return tuple(self.prior_transform(theta))

        ndim = len(self.fitting_parameters)
        self.warning("Beginning fit......")
        ndims = ndim  # two parameters

        t0 = time.time()

        res = nestle.sample(
            self.log_likelihood,
            nestle_uniform_prior,
            ndims,
            method="multi",
            npoints=self.numLivePoints,
            dlogz=self.tolerance,
            callback=nestle.print_progress,
        )

        res = t.cast(nestle.Result, res)
        t1 = time.time()

        timenestle = t1 - t0

        print(res.summary())

        self.warning("Time taken to run 'Nestle' is %s seconds", timenestle)

        self.warning("Fit complete.....")

        self._nestle_output = self.store_nestle_output(res)

    def get_samples(self, solution_idx: int) -> npt.NDArray[np.float64]:
        """Returns the samples from the fit."""
        return self._nestle_output["solution"]["samples"]

    def get_weights(self, solution_idx: int) -> npt.NDArray[np.float64]:
        """Returns the weights of the samples."""
        return self._nestle_output["solution"]["weights"]

    def get_solution(
        self,
    ) -> t.Generator[
        t.Tuple[
            int,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            t.Tuple[
                t.Tuple[t.Literal["Statistics"], float],
                t.Tuple[t.Literal["fit_params"], t.Dict[str, FitParamOutput]],
                t.Tuple[t.Literal["tracedata"], npt.NDArray[np.float64]],
                t.Tuple[t.Literal["weights"], npt.NDArray[np.float64]],
            ],
        ],
        None,
        None,
    ]:
        """Generator for solutions and their median and MAP values

        Yields
        ------

        solution_no:
            Solution number (always 0)

        map:
            Map values

        median:
            Median values

        extra:
            statistics, fit_params, tracedata, weights

        """

        names = self.fit_names
        opt_map = self.fit_values
        opt_values = self.fit_values
        for k, v in self._nestle_output["solution"]["fitparams"].items():
            # if k.endswith('_derived'):
            #     continue
            idx = names.index(k)
            opt_map[idx] = v["map"]
            opt_values[idx] = v["value"]

        yield 0, opt_map, opt_values, (
            ("Statistics", self._nestle_output["Stats"]),
            ("fit_params", self._nestle_output["solution"]["fitparams"]),
            ("tracedata", self._nestle_output["solution"]["samples"]),
            ("weights", self._nestle_output["solution"]["weights"]),
        )

    def write_optimizer(self, output: OutputGroup) -> OutputGroup:
        """Writes the optimizer to the output group."""
        opt = super().write_optimizer(output)

        # number of live points
        opt.write_scalar("num_live_points", self._nlive)
        # maximum no. of iterations (0=inf)
        opt.write_string("method", self._method)
        # search for multiple modes
        opt.write_scalar("tol", self._tol)

        return opt

    def write_fit(self, output: OutputGroup) -> OutputGroup:
        """Writes the fit to the output group."""
        fit = super().write_fit(output)

        if self._nestle_output:
            recursively_save_dict_contents_to_output(output, self._nestle_output)

        return fit

    def store_nestle_output(self, result: nestle.Result) -> NestleSolutionOutput:
        """This turns the output fron nestle into a dictionary

        Contains summary statistics and the solution.

        """
        nestle_output = {}
        nestle_output["Stats"] = {}
        nestle_output["Stats"]["Log-Evidence"] = result.logz
        nestle_output["Stats"]["Log-Evidence-Error"] = result.logzerr
        nestle_output["Stats"]["Peakiness"] = result.h

        fit_param = self.fit_names

        samples = result.samples
        weights = result.weights

        mean, cov = nestle.mean_and_cov(samples, weights)
        nestle_output["solution"] = {}
        nestle_output["solution"]["samples"] = samples
        nestle_output["solution"]["weights"] = weights
        nestle_output["solution"]["covariance"] = cov
        nestle_output["solution"]["fitparams"] = {}

        max_weight = weights.argmax()

        table_data = []

        for idx, param_name in enumerate(fit_param):
            param = {}
            param["mean"] = mean[idx]
            param["sigma"] = cov[idx]
            trace = samples[:, idx]
            q_16, q_50, q_84 = quantile_corner(
                trace, [0.16, 0.5, 0.84], weights=np.asarray(weights)
            )
            param["value"] = q_50
            param["sigma_m"] = q_50 - q_16
            param["sigma_p"] = q_84 - q_50
            param["trace"] = trace
            param["map"] = trace[max_weight]
            table_data.append((param_name, q_50, q_50 - q_16))

            nestle_output["solution"]["fitparams"][param_name] = param

        return nestle_output

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("nestle",)

    BIBTEX_ENTRIES = [
        """@misc{nestle,

        author = {Kyle Barbary},
        title = {Nestle sampling library},
        publisher = {GitHub},
        journal = {GitHub repository},
        year = 2015,
        howpublished = {https://github.com/kbarbary/nestle},
        }"""
    ]
