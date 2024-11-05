"""Base class for optimization/retrieval."""

import math
import typing as t
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from taurex import OutputSize
from taurex.cache import GlobalCache
from taurex.core import DerivedType, FittingType
from taurex.core.priors import LogUniform, Prior, PriorMode, Uniform
from taurex.data.citation import Citable
from taurex.log import Logger, disableLogging, enableLogging
from taurex.model import ForwardModel
from taurex.output import OutputGroup
from taurex.spectrum import BaseSpectrum
from taurex.types import AnyValType

SQRTPI = np.sqrt(2 * np.pi)


class FitParamOutput(t.TypedDict):
    """Dictionary for storing fit parameter output."""

    mean: float
    sigma: float
    value: float
    sigma_m: float
    sigma_p: float
    trace: float
    map: float


@dataclass
class FitParam:
    """Holds information about a fitting parameter."""

    name: str
    latex: str
    fget: t.Callable[[], float]
    fset: t.Callable[[float], None]
    mode: t.Literal["linear", "log"]
    to_fit: bool
    bounds: t.Tuple[float, float]
    prior: Prior = None

    @property
    def fit_prior(self):
        """Get prior for fitting."""
        if self.prior is not None:
            return self.prior
        if self.mode == "linear":
            return Uniform(bounds=self.bounds)
        return LogUniform(lin_bounds=self.bounds)

    @property
    def value(self) -> float:
        """Get value of parameter."""
        return self.fget()

    @value.setter
    def value(self, value: float) -> None:
        """Set value of parameter.

        Parameters
        ----------
        value : float
            Value to set parameter to

        """
        self.fset(self.fit_prior.prior(value))

    @property
    def fit_name(self) -> str:
        """Name for fitting."""
        return (
            self.name
            if self.fit_prior.priorMode is PriorMode.LINEAR
            else f"log_{self.name}"
        )

    @property
    def fit_latex(self) -> str:
        """Latex for fitting."""
        return (
            self.latex
            if self.fit_prior.priorMode is PriorMode.LINEAR
            else f"log({self.latex})"
        )

    @property
    def fit_value(self) -> float:
        """Get value of parameter considering its mode."""
        import math

        return (
            self.value
            if self.fit_prior.priorMode is PriorMode.LINEAR
            else math.log10(self.value)
        )

    @property
    def boundaries(self):
        """Get boundaries of parameter."""
        return self.fit_prior.boundaries()


@dataclass
class DerivedParam:
    """Holds information about a derived parameter."""

    name: str
    latex: str
    fget: t.Callable[[], float]
    compute: bool


def compile_params(
    fitparams: t.Dict[str, FittingType],
    driveparams: t.Dict[str, DerivedType],
    fit_priors: t.Dict[str, Prior] = None,
):
    """Compile fitting and derived parameters."""
    fitting_parameters = [
        FitParam(*params, prior=fit_priors.get(params[0]))
        for params in fitparams.values()
    ]
    derived_parameters = [DerivedParam(*params) for params in driveparams.values()]

    return fitting_parameters, derived_parameters


class Optimizer(Logger, Citable):
    """A base class that handles fitting and optimization of forward models.

    The class handles the compiling and management of fitting parameters in
    forward models, in its current form it cannot fit and requires a class
    derived from it to implement the :func:`compute_fit` function.

    """

    def __init__(
        self,
        name: str,
        observed: t.Optional[BaseSpectrum] = None,
        model: t.Optional[ForwardModel] = None,
        sigma_fraction: t.Optional[float] = 0.1,
    ) -> None:
        """Initilize optimizer.

        Parameters
        ----------
        name:
            Name of optimizer for logging

        observed:
            Observed spectrum

        model:
            Model to be optimized

        sigma_fraction:
            Fraction of weights to use in computing the error. (Default: 10%)


        """
        super().__init__(name)

        self._model_callback = None
        self._sigma_fraction = sigma_fraction
        self._fit_priors = {}
        self.avail_fit_parameters: t.Dict[str, FitParam] = {}
        self.avail_derived_parameters: t.Dict[str, DerivedParam] = {}
        self._model: ForwardModel = None
        self._observed: BaseSpectrum = None
        self.set_model(model)
        self.set_observed(observed)

    def set_model(self, model: ForwardModel) -> None:
        """Sets the model to be optimized/fit

        Parameters
        ----------
        model:
            The forward model we wish to optimize

        """
        self._model: ForwardModel = model
        self._compile_params()

    def set_observed(self, observed: BaseSpectrum) -> None:
        """Sets the observation to optimize the model to.

        Parameters
        ----------
        observed:
            Observed spectrum we will optimize to

        """
        self._observed: BaseSpectrum = observed
        if observed is not None:
            self._binner = observed.create_binner()
        self._compile_params()

    def compile_params(self) -> None:
        """Dummy, does nothing and will be depcreated"""
        import warnings

        warnings.warn(
            "compile_params is deprecated and will be removed in future versions. ",
            DeprecationWarning,
            stacklevel=2,
        )

    def _compile_params(self) -> None:
        """Compile parameters for fitting from model and observation."""
        self.info("Initializing parameters")
        model_fit, model_derive = [], []
        if self._model is not None:
            (
                model_fit,
                model_derive,
            ) = compile_params(
                self._model.fittingParameters,
                self._model.derivedParameters,
                self._fit_priors,
            )

        obs_fit, obs_deriv = [], []
        if self._observed is not None:
            obs_fit, obs_deriv = compile_params(
                self._observed.fittingParameters,
                self._observed.derivedParameters,
                self._fit_priors,
            )

        fitting_params = model_fit + obs_fit
        derived_params = model_derive + obs_deriv
        # self.fitting_priors.extend(obs_prior)
        self.avail_fit_parameters = {p.name: p for p in fitting_params}
        self.avail_derived_parameters = {p.name: p for p in derived_params}

        self.info("-------FITTING---------------")
        self.info("Fitting Parameters available:")
        for params in self.avail_fit_parameters.values():
            self.info(
                "%s, Value: %s Type: %s, Params: %s",
                params.name,
                params.fget(),
                params.fit_prior.__class__.__name__,
                params.fit_prior.params(),
            )
        self.info("-------DERIVED---------------")
        self.info("Derived Parameters available:")
        for params in self.avail_derived_parameters.values():
            self.info("%s, Value: %s", params.name, params.fget())

    @property
    def fitting_parameters(self) -> t.List[FitParam]:
        """Returns the list of fitting parameters."""
        return [f for f in self.avail_fit_parameters.values() if f.to_fit]

    @property
    def derived_parameters(self) -> t.List[DerivedParam]:
        """Returns the list of derived parameters."""
        return [f for f in self.avail_derived_parameters.values() if f.compute]

    @property
    def fitting_priors(self) -> t.List[Prior]:
        """Returns the list of fitting priors.

        This is mostly for compatibility with the old code.

        """
        return [f.fit_prior for f in self.fitting_parameters]

    def prior_transform(self, cube: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Transforms a unit cube to a prior cube.

        Employ mixins to override this method.

        """
        return np.array(
            [fit.fit_prior.sample(c) for fit, c in zip(self.fitting_parameters, cube)]
        )

    def log_likelihood(self, parameters: npt.ArrayLike) -> float:
        r"""Log likelihood function.

        Computed as:

        .. math::

            \log \mathcal{L} = -\sum_i \sigma_i \sqrt{2\pi}) - \frac{1}{2} \chi^2

        Employ mixins to override this method.

        """
        parameters = np.asarray(parameters)
        data = self._observed.spectrum
        datastd = self._observed.errorBar

        chi_t = self.chisq_trans(parameters, data, datastd)
        loglike = -np.sum(np.log(datastd * SQRTPI)) - 0.5 * chi_t
        return loglike

    def update_model(self, fit_params: t.List[float]) -> None:
        """Updates the model with new parameters

        Parameters
        ----------
        fit_params : :obj:`list`
            A list of new values to apply to the model. The list of values are
            assumed to be in the same order as the parameters given by
            :func:`fit_names`


        """
        if len(fit_params) != len(self.fitting_parameters):
            self.error(
                "Trying to update model with more fitting parameters" " than enabled"
            )
            self.error(
                f"No. enabled parameters:{len(self.fitting_parameters)}"
                f" Update length: {len(fit_params)}"
            )
            raise ValueError(
                "Trying to update model with more fitting" " parameters than enabled"
            )

        for value, param in zip(fit_params, self.fitting_parameters):
            param.value = value

    @property
    def fit_values_nomode(self) -> t.List[float]:
        """Returns a list of the current values of a fitting parameter.
        Regardless of the ``mode`` setting

        Returns
        -------
        :obj:`list`:
            List of each value of a fitting parameter

        """

        return [c.value for c in self.fitting_parameters]

    @property
    def fit_values(self) -> t.List[float]:
        """Returns a list of the current values of a fitting parameter.

        This respects the ``mode`` setting

        Returns
        -------
        :obj:`list`:
            List of each value of a fitting parameter

        """

        return [c.fit_value for c in self.fitting_parameters]

    @property
    def fit_boundaries(self):
        """

        Returns the fitting boundaries of the parameter

        Returns
        -------
        :obj:`list`:
            List of boundaries for each fitting parameter. It takes the form of
            a python :obj:`tuple` with the form
            ( ``bound_min`` , ``bound_max`` )

        """
        return [
            c[-1] if c[4] == "linear" else (math.log10(c[-1][0]), math.log10(c[-1][1]))
            for c in self.fitting_parameters
        ]

    @property
    def fit_names(self) -> t.List[str]:
        """Returns the names of the model parameters we will be fitting

        Returns
        -------
        :obj:`list`:
            List of names of parameters that will be fit

        """

        return [c.fit_name for c in self.fitting_parameters]

    @property
    def fit_latex(self) -> t.List[str]:
        """Returns the names of the parameters in LaTeX format."""
        return [c.fit_latex for c in self.fitting_parameters]

    @property
    def derived_names(self) -> t.List[str]:
        """Return names for derived parameters."""
        return [c.name for c in self.derived_parameters]

    @property
    def derived_latex(self) -> t.List[str]:
        """Returns a list of the current values of a fitting parameter."""
        return [c.latex for c in self.derived_parameters]

    @property
    def derived_values(self) -> t.List[float]:
        """Returns current values of derived parameters."""
        return [c.fget() for c in self.derived_parameters]

    def enable_fit(self, parameter: str) -> None:
        """Enables fitting of the parameter"""
        self.avail_fit_parameters[parameter].to_fit = True

    def enable_derived(self, parameter: str) -> None:
        """Enables computation of derived parameter."""
        self.avail_derived_parameters[parameter].compute = True

    def disable_fit(self, parameter: str) -> None:
        """Disables fitting of the parameter."""
        self.avail_fit_parameters[parameter].to_fit = False

    def disable_derived(self, parameter: str) -> None:
        """Disables computation of derived parameter."""
        self.avail_derived_parameters[parameter].compute = False

    def set_boundary(self, parameter: str, new_boundaries: t.List[float]) -> None:
        """Sets the boundary of the parameter."""
        self.avail_fit_parameters[parameter].bounds = new_boundaries

    def set_factor_boundary(self, parameter: str, factors: t.List[float]) -> None:
        """Sets the boundary of the parameter based on a factor of value."""
        param = self.avail_fit_parameters[parameter]
        bounds = (param.value * factors[0], param.value * factors[1])
        param.bounds = (min(bounds), max(bounds))

    def set_mode(self, parameter: str, new_mode: t.Literal["linear", "log"]) -> None:
        """Sets the fitting mode of a parameter."""
        self.avail_fit_parameters[parameter].mode = new_mode

    def set_prior(self, parameter: str, prior: Prior) -> None:
        """Sets the prior of a parameter."""
        self.avail_fit_parameters[parameter].prior = prior

    def chisq_trans(
        self,
        fit_params: npt.ArrayLike,
        data: npt.NDArray[np.float64],
        datastd: npt.NDArray[np.float64],
    ) -> float:
        """Computes the Chi-Squared of model and observation

        Computes the Chi-Squared between the forward model and
        observation. The steps taken are:
            1. Forward model (FM) is updated with :func:`update_model`
            2. FM is then computed at its native grid then binned.
            3. Chi-squared between FM and observation is computed
        """
        from taurex.exceptions import InvalidModelException

        self.update_model(fit_params)
        obs_bins = self._observed.wavenumberGrid

        try:
            _, final_model, _, _ = self._binner.bin_model(
                self._model.model(wngrid=obs_bins)
            )
        except InvalidModelException:
            return np.nan

        res = (data.ravel() - final_model.ravel()) / datastd.ravel()

        reject_nan = GlobalCache()["reject_spectral_nan"]

        if reject_nan and np.any(np.isnan(res)):
            return np.nan

        res = np.nansum(res * res)
        if res == 0:
            res = np.nan

        return res

    def compute_fit(self) -> t.Any:
        """Main compute fit function.

        Unimplemented. When inheriting this should be overwritten
        to perform the actual fit.

        Raises
        ------
        NotImplementedError
            Raised when a derived class does override this function

        """
        raise NotImplementedError

    def fit(self, output_size=OutputSize.heavy) -> t.Dict[str, AnyValType]:
        """Performs retrieval."""
        import time

        from tabulate import tabulate

        fit_names = self.fit_names

        prior_type = [p.__class__.__name__ for p in self.fitting_priors]
        args = [p.params() for p in self.fitting_priors]

        fit_values = self.fit_values
        self.info("")
        self.info("-------------------------------------")
        self.info("------Retrieval Parameters-----------")
        self.info("-------------------------------------")
        self.info("")
        self.info("Dimensionality of fit: %s", len(fit_names))
        self.info("")

        output = tabulate(
            zip(fit_names, fit_values, prior_type, args),
            headers=["Param", "Value", "Type", "Args"],
        )

        self.info("\n%s\n\n", output)
        self.info("")
        start_time = time.time()
        disableLogging()
        # Compute fit here
        self.compute_fit()

        enableLogging()
        end_time = time.time()
        self.info("Sampling time %s s", end_time - start_time)
        solution = self.generate_solution(output_size=output_size)
        self.info("")
        self.info("-------------------------------------")
        self.info("------Final results------------------")
        self.info("-------------------------------------")
        self.info("")
        self.info("Dimensionality of fit: %s", len(fit_names))
        self.info("")
        for idx, optimized_map, optimized_median, _ in self.get_solution():
            self.info("\n%s", f"---Solution {idx}------")
            output = tabulate(
                zip(fit_names, optimized_map, optimized_median),
                headers=["Param", "MAP", "Median"],
            )
            self.info("\n%s\n\n", output)
        return solution

    def write_optimizer(self, output: OutputGroup) -> OutputGroup:
        """Writes optimizer settings under the 'Optimizer' heading in an output file.

        Parameters
        ----------
        output:
            Group (or root) in output file to write to
        """
        output.write_string("optimizer", self.__class__.__name__)
        output.write_string_array("fit_parameter_names", self.fit_names)
        output.write_string_array("fit_parameter_latex", self.fit_latex)
        output.write_array(
            "fit_boundary_low",
            np.array([x.boundaries()[0] for x in self.fitting_priors]),
        )
        output.write_array(
            "fit_boundary_high",
            np.array([x.boundaries()[1] for x in self.fitting_priors]),
        )
        if len(self.derived_names) > 0:
            output.write_string_array("derived_parameter_names", self.derived_names)
            output.write_string_array("derived_parameter_latex", self.derived_latex)

        return output

    def write_fit(self, output: OutputGroup) -> OutputGroup:
        """Writes basic fitting parameters into output.

        Parameters
        ----------
        output:
            Group (or root) in output file to write to

        """
        fit = output.create_group("FitParams")
        fit.write_string("fit_format", self.__class__.__name__)
        fit.write_string_array("fit_parameter_names", self.fit_names)
        fit.write_string_array("fit_parameter_latex", self.fit_latex)
        fit.write_array(
            "fit_boundary_low", np.array([x[0] for x in self.fit_boundaries])
        )
        fit.write_array(
            "fit_boundary_high", np.array([x[1] for x in self.fit_boundaries])
        )

        # This is the last sampled value ... should not be recorded to avoid confusion.
        # fit.write_list('fit_parameter_values',self.fit_values)
        # fit.write_list('fit_parameter_values_nomode',self.fit_values_nomode)
        return output

    def generate_profiles(
        self, solution: int, binning: npt.NDArray[np.float64]
    ) -> t.Tuple[
        t.Dict[str, npt.NDArray[np.float64]], t.Dict[str, npt.NDArray[np.float64]]
    ]:
        """Generates sigma plots for profiles"""
        from taurex import mpi

        sample_list: t.List[float] = []

        if mpi.get_rank() == 0:
            sample_list = list(self.sample_parameters(solution))

        sample_list = mpi.broadcast(sample_list)

        self.debug("We all got %s", sample_list)

        self.info("------------Variance generation step------------------")

        self.info("We are sampling %s points for the profiles", len(sample_list))

        rank = mpi.get_rank()
        size = mpi.nprocs()

        enableLogging()

        self.info(
            "I will only iterate through partitioned %s "
            "points (the rest is in parallel)",
            len(sample_list) // size,
        )

        disableLogging()

        def sample_iter():
            count = 0
            for parameters, weight in sample_list[rank::size]:
                self.update_model(parameters)
                enableLogging()
                if rank == 0 and count % 10 == 0 and count > 0:
                    self.info(
                        "Progress %.3f", count * 100.0 / (len(sample_list) / size)
                    )
                disableLogging()
                yield weight
                count += 1

        return self._model.compute_error(
            sample_iter, wngrid=binning, binner=self._binner
        )

    def generate_solution(  # noqa: C901
        self, output_size=OutputSize.heavy
    ) -> t.Dict[str, AnyValType]:
        """Generates a dictionary with all solutions and other useful parameters."""
        from taurex.util.output import store_contributions

        solution_dict = {}

        self.info("Post-processing - Generating spectra and profiles")

        # Loop through each solution, grab optimized parameters and anything
        # else we want to store
        for solution, optimized_map, optimized_median, values in self.get_solution():
            enableLogging()
            self.info("Computing solution %s", solution)
            sol_values = {}

            # Include extra stuff we might want to store (provided by the child)
            for k, v in values:
                sol_values[k] = v

            # Update the model with optimized map values
            self.update_model(optimized_map)

            opt_result = self._model.model(cutoff_grid=False)  # Run the model

            sol_values["Spectra"] = self._binner.generate_spectrum_output(
                opt_result, output_size=output_size
            )

            try:
                sol_values["Spectra"]["Contributions"] = store_contributions(
                    self._binner, self._model, output_size=output_size - 3
                )
            except Exception as e:
                self.warning("Not bothering to store contributions since its broken")
                self.warning("%s ", str(e))

            # Update with the optimized median
            self.update_model(optimized_median)

            self._model.model(cutoff_grid=False)

            # Store profiles here
            sol_values["Profiles"] = self._model.generate_profiles()
            profile_dict, spectrum_dict = self.generate_profiles(
                solution, self._observed.wavenumberGrid
            )

            for k, v in profile_dict.items():
                if k in sol_values["Profiles"]:
                    sol_values["Profiles"][k].update(v)
                else:
                    sol_values["Profiles"][k] = v

            for k, v in spectrum_dict.items():
                if k in sol_values["Spectra"]:
                    sol_values["Spectra"][k].update(v)
                else:
                    sol_values["Spectra"][k] = v

            solution_dict[f"solution{solution}"] = sol_values

        if len(self.derived_names) > 0:
            # solution_dict[f'solution{solution}']['derived_params'] = {}
            # Compute derived
            for (
                solution,
                _,
                _,
                _,
            ) in self.get_solution():
                solution_dict[f"solution{solution}"]["derived_params"] = {}
                derived_dict = self.compute_derived_trace(solution)
                if derived_dict is None:
                    continue
                solution_dict[f"solution{solution}"]["derived_params"].update(
                    derived_dict
                )

        enableLogging()
        self.info("Post-processing - Complete")

        return solution_dict

    def compute_derived_trace(self, solution: int) -> t.Dict[str, AnyValType]:
        """Computes derived parameters from traces."""
        from taurex import mpi
        from taurex.util import quantile_corner

        enableLogging()

        samples = self.get_samples(solution)
        weights = self.get_weights(solution)
        len_samples = len(samples)

        rank = mpi.get_rank()

        num_procs = mpi.nprocs()

        count = 0

        derived_param = {p: ([], []) for p in self.derived_names}

        if len(self.derived_names) == 0:
            return

        self.info("Computing derived parameters......")
        disableLogging()
        for idx in range(rank, len_samples, num_procs):
            enableLogging()
            if rank == 0 and count % 10 == 0 and count > 0:
                self.info(f"Progress {idx * 100.0 / len_samples}%")
            disableLogging()

            parameters = samples[idx]
            weight = weights[idx]
            self.update_model(parameters)
            self._model.initialize_profiles()
            for p, v in zip(self.derived_names, self.derived_values):
                derived_param[p][0].append(v)
                derived_param[p][1].append(weight)

        result_dict = {}

        sorted_weights = weights.argsort()

        for param, (trace, w) in derived_param.items():
            # I cant remember why this works
            all_trace = np.array(mpi.allreduce(trace, op="SUM"))
            # I cant remember why this works
            all_weight = np.array(mpi.allreduce(w, op="SUM"))

            all_weight_sort = all_weight.argsort()

            # Sort them into the right order
            all_weight[sorted_weights] = all_weight[all_weight_sort]
            all_trace[sorted_weights] = all_trace[all_weight_sort]

            q_16, q_50, q_84 = quantile_corner(
                np.array(all_trace), [0.16, 0.5, 0.84], weights=np.array(all_weight)
            )

            mean = np.average(all_trace, weights=all_weight, axis=0)

            derived = {
                "value": q_50,
                "sigma_m": q_50 - q_16,
                "sigma_p": q_84 - q_50,
                "trace": all_trace,
                "mean": mean,
            }
            result_dict[f"{param}_derived"] = derived
        return result_dict

    def sample_parameters(
        self, solution: int
    ) -> t.Generator[t.Tuple[npt.NDArray[np.float64], float], None, None]:
        """Read traces and weights and return a random ``sigma_fraction`` sample

        Parameters
        ----------
        solution:
            a solution output from sampler

        Yields
        ------
        traces: :obj:`array`
            Traces of a particular sample

        weight: float
            Weight of sample

        """
        from taurex.util import random_int_iter

        samples = self.get_samples(solution)
        weights = self.get_weights(solution)

        iterator = random_int_iter(samples.shape[0], self._sigma_fraction)
        for x in iterator:
            w = weights[x] + 1e-300

            yield samples[x, :], w

    def get_solution(
        self,
    ) -> t.Generator[
        t.Tuple[
            int,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            t.Dict[str, AnyValType],
        ],
        None,
        None,
    ]:
        """Generator for solutions.

        ** Requires implementation **

        Generator for solutions and their
        median and MAP values

        Yields
        ------

        solution_no:
            Solution number

        map:
            Map values

        median:
            Median values

        extra:
            List of tuples of extra information to store.
            Must be of form ``(name, data)``



        """
        raise NotImplementedError

    def get_samples(self, solution_id: int) -> npt.NDArray[np.float64]:
        """Get traces for a particular solution."""
        raise NotImplementedError

    def get_weights(self, solution_id: int) -> npt.NDArray[np.float64]:
        """Get weights for a particular solution."""
        raise NotImplementedError

    def write(self, output: OutputGroup) -> None:
        """Creates  'Optimizer' and writes output

        Parameters
        ----------
        output :
            Group (or root) in output file to write to



        """
        opt = output.create_group("Optimizer")
        self.write_optimizer(opt)

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for optimizer."""
        raise NotImplementedError
