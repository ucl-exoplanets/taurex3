"""This module relates to defining fitting parameters in TauREx3."""

import typing as t
from functools import partial, wraps

T = t.TypeVar("T")
"""Generic type."""

P = t.TypeVar("P")


@t.runtime_checkable
class FittingCallable(t.Protocol):
    """Protocol for fitting callables."""

    param_name: str
    """Parameter name."""

    param_latex: str
    """Parameter name in latex."""

    default_fit: bool
    """Fit by default?"""

    default_bounds: t.List[float]
    """Default fitting boundaries."""

    default_mode: t.Literal["linear", "log"]
    """Default fitting mode."""

    decorated: t.Literal["fitparam"] = "fitparam"
    """Decorated type."""

    fget: t.Callable[[], float]
    """Getter."""

    fset: t.Callable[[float], None]
    """Setter."""

    fdel: t.Optional[t.Callable[[], None]] = None

    doc: t.Optional[str] = None
    """Docstring."""

    def __call__(self) -> float:
        """Callable protocol."""
        ...


@t.runtime_checkable
class DerivedCallable(t.Protocol):
    """Protocol for derived callables."""

    param_name: str
    """Parameter name."""

    param_latex: str
    """Parameter name in latex."""

    compute: bool
    """Compute by default?"""

    fget: t.Callable[[], float]
    """Getter."""

    decorated: t.Literal["derivedparam"] = "derivedparam"
    """Decorated type."""

    def __call__(self) -> float:
        """Callable protocol."""
        ...


# class FittingType(t.TypedDict):
#     """Fitting type."""

#     param_name: str
#     """Parameter name."""

#     param_latex: str
#     """Parameter name in latex."""

#     fget: t.Callable[[], float]
#     """Getter."""

#     fset: t.Callable[[float], None]
#     """Setter."""

#     default_fit: bool
#     """Fit by default?"""

#     default_bounds: t.List[float]
#     """Default fitting boundaries."""

#     default_mode: t.Literal["linear", "log"]
#     """Default fitting mode."""


# class DerivedType(t.TypedDict):
#     """Derived type."""

#     param_name: str
#     """Parameter name."""

#     param_latex: str
#     """Parameter name in latex."""

#     fget: t.Callable[[], float]
#     """Getter."""

#     compute: bool
#     """Compute by default?"""


class FitPropertyType:
    fget: FittingCallable
    fset: t.Callable[[T], None]


class DerivedPropertyType:
    fget: DerivedCallable


FittingType = t.Tuple[
    str,
    str,
    t.Callable[[], float],
    t.Callable[[float], None],
    t.Literal["linear", "log"],
    bool,
    t.Tuple[float, float],
]

DerivedType = t.Tuple[str, str, t.Callable[[], float], bool]


def fitparam(
    f: t.Optional[t.Callable[[float], T]] = None,
    param_name: t.Optional[str] = None,
    param_latex: t.Optional[str] = None,
    default_mode: t.Optional[t.Literal["linear", "log"]] = "linear",
    default_fit: t.Optional[bool] = False,
    default_bounds: t.Optional[t.List[float]] = None,
) -> FittingCallable:
    """Decorator to mark fittable parameters.

    A decorator used in conjunction with :class:`Fittable` to inform which
    parameters can be fit and its properties. On its own it acts like the
    ``property`` decorator. When used within a :class:`Fittable` class it
    serves to tag a property as able to fit and allows the
    class to compile all parameters that can be fit.

    Its usage is simple, simply wrap a method and define its properties:

    .. code-block:: python

        class Foo(Fittable):

            @fitparam(param_name='foobar',param_latex='$Foo^{bar}$')
            def bar(self):
                return 'Foobar'

            @bar.setter
            def bar(self,value):
                self.value = value


    Parameters
    ----------
    f: function
        Function being passed. Automatically done when used as a decorator

    param_name: str
        Nicer name of the parameter. Referenced by the optimizer.

    param_latex: str
        Latex version of the parameter name, useful in plotting and
        making figures

    default_mode: ``linear`` or ``log``
        Defines how the optimizer should read and write the parameter.
        ``linear`` reads/write everything as is.
        ``log`` informs the optimizer to transform from native->log space
        when read and to transfrom log->native when
        writing. This also applies to the boundaries

    default_fit: bool
        Whether this is included in the fit without the user explicity saying
        so (Default: False)

    default_bounds: :obj:`list`
        Default minimum and maximum fitting boundary. Must always be defined
        in linear space


    Raises
    ------
    ValueError
        If no parameter name is given

    """
    default_bounds = default_bounds or [0.0, 1.0]
    if f is None:
        return partial(
            fitparam,
            param_name=param_name,
            param_latex=param_latex,
            default_mode=default_mode,
            default_fit=default_fit,
            default_bounds=default_bounds,
        )

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        return f(self, *args, **kwargs)

    if param_name is None:
        raise ValueError("Fitting parameter must have a name")

    wrapper.param_name = param_name

    wrapper.param_latex = param_latex
    if wrapper.param_latex is None:
        wrapper.param_latex = param_name
    wrapper.default_fit = default_fit
    wrapper.default_bounds = default_bounds
    wrapper.default_mode = default_mode
    wrapper.decorated = "fitparam"
    pwrap = property(wrapper)
    wrapper.__doc__ = str(f.__doc__)
    return pwrap


def derivedparam(
    f: t.Optional[t.Callable[[], float]] = None,
    param_name: t.Optional[str] = None,
    param_latex: t.Optional[str] = None,
    compute: t.Optional[bool] = False,
) -> DerivedCallable:
    """Dectorator to mark derivable parameters.

    A decorator used in conjunction with :class:`Fittable` to inform which
    parameters should be derived during retrieval. This allows for posteriors
    of parameters such as log(g) and mu



    Parameters
    ----------
    f: function
        Function being passed. Automatically done when used as a decorator

    param_name: str
        Nicer name of the parameter. Referenced by the optimizer.

    param_latex: str
        Latex version of the parameter name, useful in plotting and
        making figures

    compute: bool
        By default, is this computed?

    """

    if f is None:
        return partial(
            derivedparam,
            param_name=param_name,
            param_latex=param_latex,
            compute=compute,
        )

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        return f(self, *args, **kwargs)

    if param_name is None:
        raise ValueError("Derived parameter must have a name")

    wrapper.param_name = param_name

    wrapper.param_latex = param_latex
    if param_latex is None:
        wrapper.param_latex = param_name
    wrapper.compute = compute
    wrapper.decorated = "derivedparam"
    pwrap = property(wrapper)
    wrapper.__doc__ = str(f.__doc__)
    return pwrap


class Fittable:
    """A class that manages fitting parameters.

    Not really used on its own it should really be inherited from to be used properly.
    It also provides class with the ability to read and write
    fitting parameters using their params names, for example,
    if we create a class like this:

    .. code-block:: python

        class Foo(Fittable):

            def __init__(self):
                self.value = 10

            @fitparam(param_name='foobar',param_latex='$Foo^{bar}$',default_bounds=[1,12])
            def bar(self):
                return self.value

            @bar.setter
            def bar(self,value):
                self.value = value

    We can read and write data in the standard python way like so:

    >>> foo = Foo()
    >>> foo.bar
    10
    >>> foo.bar = 20
    >>> foo.bar
    20

    but we also get this functionality for free:

    >>> foo['foobar']
    20
    >>> foo['foobar'] = 30
    >>> foo['foobar']
    30


    """

    def __init__(self) -> None:
        """Initialize the class."""
        self._param_dict: t.Dict[str, FittingType] = {}
        self._derived_dict: t.Dict[str, DerivedType] = {}

        self.compile_fitparams()

    def add_fittable_param(
        self,
        param_name: str,
        param_latex: str,
        fget: t.Callable[[], float],
        fset: t.Callable[[float], None],
        default_mode: t.Literal["linear", "log"],
        default_fit: bool,
        default_bounds: t.Tuple[float, float],
    ) -> None:
        """Adds a fittable parameter to the internal dictionary.


        Used during init to add all :func:`fitparam` decorated methods
        and can also be utilized by a user to manually add new fitting parameters.
        This is useful for giving fitting parameters names that depend on certain
        attributes (e.g. molecule name in a gas profile
        see
        :class:`~taurex.data.profiles.chemistry.gas.constantgas.ConstantGas`)
        or when converting lists into fitting parameters
        (e.g. Normalization factor in light curves
        see:
        :class:`~taurex.model.lightcurve.lightcurve.LightCurveModel` )

        Parameters
        ----------
        param_name: str
            Nicer name of the parameter. Referenced by the optimizer.

        param_latex: str
            Latex version of the parameter name, useful in plotting and making
            figures

        fget: function
            a function that returns the value of the parameter

        fset: function
            a function the writes the value of the parameter

        default_mode: ``linear`` or ``log``
            Defines how the optimizer should read and write the parameter.
            ``linear`` reads/write everything as is.
            ``log`` informs the optimizer to transform from native->log space
            when read and to transfrom log->native when writing. This also
            applies to the boundaries

        default_fit: bool
            Whether this is included in the fit without the user explicity
            saying so (Default: False)

        default_bounds: :obj:`list`
            Default minimum and maximum fitting boundary. Must always be
            defined in the native space

        """
        if param_name in self._param_dict:
            raise AttributeError(f"param name {param_name} already exists")

        self._param_dict[param_name] = (
            param_name,
            param_latex,
            fget.__get__(self),
            fset.__get__(self),
            default_mode,
            default_fit,
            default_bounds,
        )

    def add_derived_param(
        self,
        param_name: str,
        param_latex: str,
        fget: t.Callable[[], float],
        compute: bool,
    ) -> None:
        if param_name in self._param_dict:
            raise AttributeError(f"derived param name {param_name} already exists")

        self._derived_dict[param_name] = (
            param_name,
            param_latex,
            fget.__get__(self),
            compute,
        )

    def compile_fitparams(self) -> None:
        """Loops through and finds all fitting parameters.

        These are defined through the decorator in the class and adds
        it to the internal dictionary
        """
        for fitparams in self.find_fitparams():
            get_func = fitparams.fget
            set_func = fitparams.fset
            param_name = get_func.param_name
            param_latex = get_func.param_latex
            def_mode = get_func.default_mode
            def_fit = get_func.default_fit
            def_bounds = get_func.default_bounds
            self.add_fittable_param(
                param_name,
                param_latex,
                get_func,
                set_func,
                def_mode,
                def_fit,
                def_bounds,
            )

        for derivedparam in self.find_derivedparams():
            get_func = derivedparam.fget
            param_name = get_func.param_name
            param_latex = get_func.param_latex
            compute = get_func.compute

            self.add_derived_param(param_name, param_latex, get_func, compute)

    def modify_bounds(self, parameter: str, new_bounds: t.Tuple[float, float]):
        """Modifies the fitting boundary of a parameter

        Parameters
        ----------
        parameter : str
            Name of parameter (given by ``param_name`` in :func:`fitparam`)

        new_bounds : :obj:`list`
            New minimum and maximum fitting boundaries.

        """
        name, latex, fget, fset, mode, to_fit, bounds = self._param_dict[parameter]

        bounds = new_bounds

        self._param_dict[parameter] = name, latex, fget, fset, mode, to_fit, bounds

    def __getitem__(self, key: str) -> float:
        """Returns the value of a fitting parameter."""
        param = self._param_dict[key]

        return param[2]()

    def __setitem__(self, key: str, value: float) -> None:
        return self._param_dict[key][3](value)

    def find_fitparams(self) -> t.Generator[FitPropertyType, None, None]:
        """Finds and returns fitting parameters.

        Yields
        ------
        method : function
            class method that is defined with the :func:`fitparam` decorator
        """
        for klass in self.__class__.mro():
            for method in klass.__dict__.values():
                if hasattr(method, "fget"):
                    prop = method.fget
                    if hasattr(prop, "decorated"):
                        if prop.decorated == "fitparam":
                            yield method

    def find_derivedparams(self) -> t.Generator[DerivedPropertyType, None, None]:
        """Finds and returns fitting parameters.

        Yields
        ------
        method : function
            class method that is defined with the :func:`derivedparam` decorator
        """
        for klass in self.__class__.mro():
            for method in klass.__dict__.values():
                if hasattr(method, "fget"):
                    prop = method.fget
                    if hasattr(prop, "decorated"):
                        if prop.decorated == "derivedparam":
                            yield method

    def fitting_parameters(self) -> t.Dict[str, FittingType]:
        """Returns all fitting parameters found as a dictionary.

        Returns
        -------
        params : :obj:`dict`
            Dictionary with key as the parameter name (``param_name``)
            and value as a tuple with:
                * parameter name
                * parameter name in Latex form
                * get function
                * set function
                * fitting scale
                * fit as default
                * fitting boundaries

        """
        return self._param_dict

    def derived_parameters(self) -> t.Dict[str, DerivedType]:
        """Returns all derived fitting parameters."""
        return self._derived_dict
