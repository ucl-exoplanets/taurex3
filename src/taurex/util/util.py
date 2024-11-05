"""General utility functions."""
import re
import typing as t

import numpy as np
import numpy.typing as npt
from astropy import units as u

from taurex.output.output import OutputGroup
from taurex.types import AnyValType, ScalarType

mass = {
    "H": 1.00794,
    "He": 4.002602,
    "Li": 6.941,
    "Be": 9.012182,
    "B": 10.811,
    "C": 12.011,
    "N": 14.00674,
    "O": 15.9994,
    "F": 18.9984032,
    "Ne": 20.1797,
    "Na": 22.989768,
    "Mg": 24.3050,
    "Al": 26.981539,
    "Si": 28.0855,
    "P": 30.973762,
    "S": 32.066,
    "Cl": 35.4527,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955910,
    "Ti": 47.88,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.93805,
    "Fe": 55.847,
    "Co": 58.93320,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.39,
    "Ga": 69.723,
    "Ge": 72.61,
    "As": 74.92159,
    "Se": 78.96,
    "Br": 79.904,
    "Kr": 83.80,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.90585,
    "Zr": 91.224,
    "Nb": 92.90638,
    "Mo": 95.94,
    "Tc": 98,
    "Ru": 101.07,
    "Rh": 102.90550,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.411,
    "In": 114.82,
    "Sn": 118.710,
    "Sb": 121.757,
    "Te": 127.60,
    "I": 126.90447,
    "Xe": 131.29,
    "Cs": 132.90543,
    "Ba": 137.327,
    "La": 138.9055,
    "Ce": 140.115,
    "Pr": 140.90765,
    "Nd": 144.24,
    "Pm": 145,
    "Sm": 150.36,
    "Eu": 151.965,
    "Gd": 157.25,
    "Tb": 158.92534,
    "Dy": 162.50,
    "Ho": 164.93032,
    "Er": 167.26,
    "Tm": 168.93421,
    "Yb": 173.04,
    "Lu": 174.967,
    "Hf": 178.49,
    "Ta": 180.9479,
    "W": 183.85,
    "Re": 186.207,
    "Os": 190.2,
    "Ir": 192.22,
    "Pt": 195.08,
    "Au": 196.96654,
    "Hg": 200.59,
    "Tl": 204.3833,
    "Pb": 207.2,
    "Bi": 208.98037,
    "Po": 209,
    "At": 210,
    "Rn": 222,
    "Fr": 223,
    "Ra": 226.0254,
    "Ac": 227,
    "Th": 232.0381,
    "Pa": 213.0359,
    "U": 238.0289,
    "Np": 237.0482,
    "Pu": 244,
    "Am": 243,
    "Cm": 247,
    "Bk": 247,
    "Cf": 251,
    "Es": 252,
    "Fm": 257,
    "Md": 258,
    "No": 259,
    "Lr": 260,
    "Rf": 261,
    "Db": 262,
    "Sg": 263,
    "Bh": 262,
    "Hs": 265,
    "Mt": 266,
    "e-": 5.4857990907e-4,
}

ElementType = t.Dict[str, int]


def calculate_weight(chem: str) -> float:
    """Compute the molecular weight of a molecule in amu.

    Parameters
    ----------
    chem : str
        Molecule name e.g. H2O, CO2, CH4, etc.

    Returns
    -------
    float
        Molecular weight in amu

    """
    s = split_molecule_elements(chem)
    compoundweight = 0.0
    for element, count in s.items():
        compoundweight += mass[element] * count
    return compoundweight


# def split_molecule_elements(chem):
#     s = re.findall('([A-Z][a-z]?)([0-9]*)', chem)
#     return s


def tokenize_molecule(molecule: str) -> list[str]:
    """Tokenize a molecule string into its elements and numbers."""
    import re

    return re.findall(r"[A-Z][a-z]?|\d+|.", molecule)


def merge_elements(
    elem1: ElementType, elem2: ElementType, factor: t.Optional[int] = 1
) -> ElementType:
    """Merge two element dictionaries."""
    return {
        elem: elem1.get(elem, 0) + elem2.get(elem, 0) * factor
        for elem in set(elem1) | set(elem2)
    }


def split_molecule_elements(  # noqa: C901
    molecule: t.Optional[str] = None, tokens: t.Optional[t.List[str]] = None
) -> ElementType:
    """Split a molecule string into its elements and numbers.

    For example when run with H2O:

    >>> split_molecule_elements('H2O')
    {'H': 2, 'O': 1}

    Parameters
    ----------
    molecule : str, optional
        Molecule string to split

    tokens : list[str], optional
        List of presplit tokens.

    Returns
    -------
    dict[str, int]
        Dictionary of elements and their counts

    """
    from taurex.util import mass

    elems = {}

    if molecule:
        tokens = tokenize_molecule(molecule)

    length = 0

    while length < len(tokens):
        token = tokens[length]

        if token in mass:
            if token not in elems:
                elems[token] = 0
            try:
                peek = int(tokens[length + 1])

                length += 1
            except IndexError:
                peek = 1
            except ValueError:
                peek = 1
            elems[token] += peek
        elif token in "{([":  # noqa: S105
            length += 1
            sub_elems, moved = split_molecule_elements(tokens=tokens[length:])
            length += moved
            try:
                peek = int(tokens[length + 1])
                length += 1
            except IndexError:
                peek = 1
            except ValueError:
                peek = 1
            elems = merge_elements(elems, sub_elems, peek)
        elif token in "}])":  # noqa: S105
            return elems, length
        length += 1

    return elems


def sanitize_molecule_string(molecule: str) -> str:
    """Cleans a molecule string to match up with molecule naming in TauREx3.

    For example:

    >>> sanitize_molecule_string('H2O')
    'H2O'
    >>> sanitize_molecule_string('H2-16O')
    'H2O'


    Parameters
    ----------
    molecule: str
        Molecule to sanitize

    Returns
    -------
    str:
        Sanitized name

    """
    return "".join(["".join(s) for s in re.findall("([A-Z][a-z]?)([0-9]*)", molecule)])


_mol_latex = {
    "HE": "He",
    "H2": "H$_2$",
    "N2": "N$_2$",
    "O2": "O$_2$",
    "CO2": "CO$_2$",
    "CH4": "CH$_4$",
    "CO": "CO",
    "NH3": "NH$_3$",
    "H2O": "H$_2$O",
    "C2H2": "C$_2$H$_2$",
    "HCN": "HCN",
    "H2S": "H$_2$S",
    "SIO2": "SiO$_2$",
    "SO2": "SO$_2$",
}
"""Latex versions of molecule names"""


def get_molecular_weight(gasname: str) -> float:
    """For a given molecule return the molecular weight in kg

    Parameters
    ----------
    gasname : str
        Name of molecule

    Returns
    -------
    float :
        molecular weight in amu or 0 if not found

    """
    from taurex.constants import AMU

    mu = calculate_weight(gasname)

    return mu * AMU


# TODO: Generalize this to any molecule.
def molecule_texlabel(gasname: str) -> str:
    """For a given molecule return its latex form


    Parameters
    ----------
    gasname : str
        Name of molecule

    Returns
    -------
    str :
        Latex form of the molecule or just the passed name if not found


    """
    gasname = gasname

    try:
        return _mol_latex[gasname]
    except KeyError:
        return gasname


def bindown(
    original_bin: npt.NDArray, original_data: npt.NDArray, new_bin: npt.NDArray
) -> npt.NDArray:
    """This method quickly bins down by taking the mean.

    The numpy histogram function is exploited to do this quickly.
    This is prone to nans if no data is present in the bin.

    Parameters
    ----------
    original_bin: :obj:`numpy.array`
        The original bins for the that we want to bin down

    original_data: :obj:`numpy.array`
        The associated data that will be averaged along the new bins

    new_bin: :obj:`numpy.array`
        The new binnings we want to use (must have less points than the original)

    Returns
    -------
    :obj:`array`
        Binned mean of ``original_data``


    """
    import numpy as np

    # print(original_bin.shape,original_data.shape)
    # if last_point is None:
    #    last_point = new_bin[-1]*2
    # calc_bin = np.append(new_bin,last_point)
    # return(np.histogram(original_bin, calc_bin, weights=original_data)[0] /
    #          np.histogram(original_bin,calc_bin)[0])

    filter_lhs = np.zeros(new_bin.shape[0] + 1)
    filter_lhs[0] = new_bin[0]
    filter_lhs[0] -= (new_bin[1] - new_bin[0]) / 2
    filter_lhs[-1] = new_bin[-1]
    filter_lhs[-1] += (new_bin[-1] - new_bin[-2]) / 2
    filter_lhs[1:-1] = (new_bin[1:] + new_bin[:-1]) / 2
    axis = len(original_data.shape) - 1
    if axis:
        digitized = np.digitize(original_bin, filter_lhs, right=True)
        axis = len(original_data.shape) - 1
        bin_means = [
            original_data[..., digitized == i].mean(axis=axis)
            for i in range(1, len(filter_lhs))
        ]
        return np.column_stack(bin_means)
    return (
        np.histogram(original_bin, filter_lhs, weights=original_data)[0]
        / np.histogram(original_bin, filter_lhs)[0]
    )


def movingaverage(a: npt.NDArray, n: t.Optional[int] = 3) -> npt.NDArray:
    """Computes moving average given an array and window size.

    Parameters
    ----------
    a : :obj:`array`
        Array to compute average

    n : int
        Averaging window

    Returns
    -------
    :obj:`array`
        Resultant array

    """
    import numpy as np

    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def quantile_corner(
    x: npt.NDArray,
    q: t.Union[npt.NDArray, float],
    weights: t.Optional[t.Union[float, npt.NDArray]] = None,
) -> npt.NDArray:
    """Compute quantiles from an array with weighting.

    * Taken from corner.py
    __author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
    __copyright__ = "Copyright 2013-2015 Daniel Foreman-Mackey"

    Like numpy.percentile, but:

    * Values of q are quantiles [0., 1.] rather than percentiles [0., 100.]
    * scalar q not supported (q must be iterable)
    * optional weights on x

    Parameters
    ----------

    x : :obj:`array`
        Input array or object that can be converted to an array.

    q : :obj:`array` or float
        Percentile or sequence of percentiles to compute, which
        must be between 0 and 1 inclusive.

    weights : :obj:`array` or float , optional
        Weights on x

    Returns
    -------
    percentile : scalar or ndarray


    """
    import numpy as np

    if weights is None:
        return np.percentile(x, [100.0 * qi for qi in q])
    else:
        idx = np.argsort(x)
        xsorted = x[idx]
        cdf = np.add.accumulate(weights[idx])
        cdf /= cdf[-1]
        return np.interp(q, cdf, xsorted).tolist()


def loadtxt2d(intext: str) -> npt.NDArray:
    """Wraps loadtext.

    Returns a 2d array or 1d array depending on the input text.

    Parameters
    ----------
    intext : str
        Input text

    Returns
    -------
    :obj:`array`
        2d array or 1d array

    """
    try:
        return np.loadtxt(intext, ndmin=2)
    except Exception:
        return np.loadtxt(intext)


def read_error_line(line: str) -> t.Tuple[str, float, float]:
    """Reads line from multinest"""
    print("_read_error_line -> line>", line)
    name, values = line.split("   ", 1)
    print("_read_error_line -> name>", name)
    print("_read_error_line -> values>", values)
    name = name.strip(": ").strip()
    values = values.strip(": ").strip()
    v, error = values.split(" +/- ")
    return name, float(v), float(error)


def read_error_into_dict(line: str, d: t.Dict[str, float]) -> None:
    """Reads multinest error into dict."""
    name, v, error = read_error_line(line)
    d[name.lower()] = v
    d["%s error" % name.lower()] = error


def read_table(
    txt: str, d: t.Optional[t.Dict[str, float]] = None, title: t.Optional[str] = None
):
    """Reads a table into a dictionary from multinest outputs."""
    from io import StringIO

    import numpy as np

    if title is None:
        title, table = txt.split("\n", 1)
    else:
        table = txt
    header, table = table.split("\n", 1)
    data = loadtxt2d(StringIO(table))
    if d is not None:
        d[title.strip().lower()] = data
    if len(data.shape) == 1:
        data = np.reshape(data, (1, -1))
    return data


def decode_string_array(f):
    """Helper to decode strings from hdf5."""
    sl = list(f)
    return [s[0].decode("utf-8") for s in sl]


OutputItem = t.Union[
    float, int, np.int64, np.float64, np.ndarray, str, t.List, t.Tuple, dict
]


def recursively_save_dict_contents_to_output(
    output: OutputGroup, dic: t.Dict[str, OutputItem]
):
    """Will recursive write a dictionary into output.

    Parameters
    ----------
    output:
        Group (or root) in output file to write to

    dic: dict
        Dictionary we want to write

    Raises
    ------
    ValueError
        If item is not a supported type

    """
    for key, item in dic.items():
        try:
            store_thing(output, key, item)
        except TypeError as e:
            raise ValueError("Cannot save %s type" % type(item)) from e


def store_thing(output: OutputGroup, key: str, item: OutputItem) -> None:  # noqa: C901
    """Stores a single item into output.

    Parameters
    ----------
    output:
        Group (or root) in output file to write to
    key: str
        Name of item
    item:
        Item to store

    Raises
    ------
    TypeError
        If item is not a supported type

    """
    if isinstance(
        item,
        (
            float,
            int,
            np.int64,
            np.float64,
        ),
    ):
        output.write_scalar(key, item)
    elif isinstance(item, (np.ndarray,)):
        output.write_array(key, item)
    elif isinstance(item, (str,)):
        output.write_string(key, item)
    elif isinstance(
        item,
        (
            list,
            tuple,
        ),
    ):
        if isinstance(item, tuple):
            item = list(item)
        if True in [isinstance(x, str) for x in item]:
            output.write_string_array(key, item)
        else:
            try:
                output.write_array(key, np.array(item))

            except TypeError:
                for idx, val in enumerate(item):
                    new_key = f"{key}{idx}"
                    store_thing(output, new_key, val)

    elif isinstance(item, dict):
        group = output.create_group(key)
        recursively_save_dict_contents_to_output(group, item)
    else:
        raise TypeError


def weighted_avg_and_std(
    values: npt.ArrayLike, weights: npt.ArrayLike, axis: t.Optional[int] = None
) -> t.Tuple[AnyValType, AnyValType]:
    """Computes weight average and standard deviation.

    Parameters
    ----------
    values : :obj:`array`
        Input array

    weights : :obj:`array`
        Must be same shape as ``values``


    axis : int , optional
        axis to perform weighting

    """
    import numpy as np

    average = np.average(values, weights=weights, axis=axis)
    variance = np.average(
        (values - average) ** 2, weights=weights, axis=axis
    )  # Fast and numerically precise
    return (average, np.sqrt(variance))


def random_int_iter(total: int, fraction: t.Optional[float] = 1.0) -> t.Iterator[int]:
    """Iterator to randomly sample integers up to a total number.

    Fraction is the fraction of total to sample. For example
    if total = 100 and fraction = 0.1 then 10 random integers
    will be sampled between 0 and 99.



    Parameters
    ----------
    total : int
        Maximum number
    fraction : float
        Fraction of total to sample

    Yields
    ------
    int
        Random integer

    """
    import random

    n_points = int(total * fraction)

    samples = random.sample(range(total), n_points)
    yield from samples


def compute_bin_edges(spectral_grid: npt.NDArray) -> t.Tuple[npt.NDArray, npt.NDArray]:
    """Computes bin edges from a spectral grid.

    Parameters
    ----------
    spectral_grid : :obj:`array`
        Spectral grid

    Returns
    -------
    :obj:`array`
        Bin edges
    :obj:`array`
        Bin widths

    """
    import numpy as np

    diff = np.diff(spectral_grid) / 2
    edges = np.concatenate(
        [
            [spectral_grid[0] - (spectral_grid[1] - spectral_grid[0]) / 2],
            spectral_grid[:-1] + diff,
            [(spectral_grid[-1] - spectral_grid[-2]) / 2 + spectral_grid[-1]],
        ]
    )
    return edges, np.abs(np.diff(edges))


def clip_native_to_wngrid(
    native_grid: npt.NDArray, spectral: npt.NDArray
) -> npt.NDArray:
    """Clips native grid values to a different spectral grid.

    Parameters
    ----------
    native_grid : :obj:`array`
        Native spectral grid

    spectral : :obj:`array`
        spectral grid

    Returns
    -------
    :obj:`array`
        Clipped native spectral grid

    """
    min_spectral = spectral.min()
    max_spectral = spectral.max()
    # Compute the maximum width
    wnwidths = compute_bin_edges(spectral)[-1]
    wn_min = min_spectral - wnwidths.max()
    wn_max = max_spectral + wnwidths.max()

    native_filter = (native_grid >= wn_min) & (native_grid <= wn_max)
    return native_grid[native_filter]


def wnwidth_to_wlwidth(wngrid: npt.NDArray, wnwidth: npt.NDArray) -> npt.NDArray:
    """Converts a wavenumber width to wavelength width and vice versa.

    Given a spectral grid and its associated spectral bin widths, this
    function will convert the wavenumber widths to wavelength widths and
    vice versa.

    The formula used is:

    .. math::
        \\Delta \\lambda = \\frac{10000 \\Delta \\nu}{\\nu^2}



    Parameters
    ----------
    wngrid : :obj:`array`
        Wavenumber grid in :math:`cm^{-1}`

    wnwidth : :obj:`array`
        Wavenumber width in :math:`cm^{-1}`

    Returns
    -------
    :obj:`array`
        Wavelength width in :math:`\\mu m`

    """
    return 10000 * wnwidth / (wngrid**2)


def class_from_keyword(keyword, class_filter=None):
    from ..parameter.classfactory import ClassFactory

    cf = ClassFactory()

    combined_classes = []
    if class_filter is None:
        combined_classes = (
            list(cf.temperatureKlasses)
            + list(cf.pressureKlasses)
            + list(cf.chemistryKlasses)
            + list(cf.gasKlasses)
            + list(cf.planetKlasses)
            + list(cf.starKlasses)
            + list(cf.modelKlasses)
            + list(cf.contributionKlasses)
        )
    else:
        if hasattr(class_filter, "__len__"):
            for x in class_filter:
                combined_classes += list(cf.list_from_base(x))
        else:
            combined_classes = list(cf.list_from_base(class_filter))

    for x in combined_classes:
        try:
            if keyword in x.input_keywords():
                return x
        except NotImplementedError:
            continue

    return None


def class_for_name(class_name: str):
    """Converts a string to a class.

    Searches TauREx3 registry of classes (including plugins) for name.



    Parameters
    ----------
    class_name : str
        Name of class

    """
    from ..parameter.classfactory import ClassFactory

    cf = ClassFactory()

    combined_classes = (
        list(cf.temperatureKlasses)
        + list(cf.pressureKlasses)
        + list(cf.chemistryKlasses)
        + list(cf.gasKlasses)
        + list(cf.planetKlasses)
        + list(cf.starKlasses)
        + list(cf.modelKlasses)
        + list(cf.contributionKlasses)
    )

    try:
        class_name = class_name.decode()
    except (UnicodeDecodeError, AttributeError):
        pass

    combined_classes_name = [c.__name__ for c in combined_classes]

    if class_name in combined_classes_name:
        return combined_classes[combined_classes_name.index(class_name)]
    else:
        raise Exception(f"Class of name {class_name} does not exist")


def create_grid_res(
    resolution: ScalarType, spectral_min: ScalarType, spectral_max: ScalarType
) -> npt.NDArray:
    """Creates a grid with a given resolution.

    Resolution is defined as :math:`R = \\frac{\\lambda}{\\Delta \\lambda}`

    Parameters
    ----------
    resolution : float
        Resolution to use
    spectral_min : float
        Minimum wavelength
    spectral_max : float
        Maximum wavelength

    Returns
    -------
    :obj:`array`
        Grid with resolution and spectral bin widths

    """
    #
    # R = l/Dl
    # l = (l-1)+Dl/2 + (Dl-1)/2
    #
    # --> (R - 1/2)*Dl = (l-1) + (Dl-1)/2
    #
    #
    spectral_list = []
    width_list = []
    wave = spectral_min
    width = wave / resolution

    while wave < spectral_max:
        width = wave / (resolution - 0.5) + width / 2 / (resolution - 0.5)
        wave = resolution * width
        width_list.append(width)
        spectral_list.append(wave)

    return np.array((spectral_list, width_list)).T


def conversion_factor(from_unit: str, to_unit: str) -> float:
    """Determine conversion from one unit to another.

    Parameters
    ----------
    from_unit : :class:`~astropy.units.Unit`
        Unit to convert from

    to_unit : :class:`~astropy.units.Unit`
        Unit to convert to

    Returns
    -------
    float
        Conversion factor.

    """
    try:
        from_conv = u.Unit(from_unit)
    except ValueError:
        from_conv = u.Unit(from_unit, format="cds")

    try:
        to_conv = u.Unit(to_unit)
    except ValueError:
        to_conv = u.Unit(to_unit, format="cds")

    return from_conv.to(to_conv)


def compute_dz(altitude: npt.NDArray) -> npt.NDArray:
    dz = np.zeros_like(altitude)
    dz[:-1] = np.diff(altitude)
    dz[-1] = altitude[-1] - altitude[-2]

    return dz


def has_duplicates(arr: npt.ArrayLike) -> bool:
    """Checks if an array has duplicates."""
    return len(arr) != len(set(arr))


def find_closest_pair(arr, value) -> (int, int):
    """Will find the indices that lie to the left and right of the value.

    `arr[left] <= value <= arr[right]`

    If the value is less than the array minimum then it will
    always return left=0 and right=1

    If the value is above the maximum

    Parameters
    ----------
    arr: :obj:`array`
        Array to search, must be sorted

    value: float
        Value to find in array


    Returns
    -------
    left: int

    right: int

    """

    right = arr.searchsorted(value)
    right = max(min(arr.shape[0] - 1, right), 1)

    left = right - 1
    left = max(0, left)

    return left, right


def ensure_string_utf8(val: str) -> str:
    """Ensures a string is utf8 encoded."""
    output = val

    try:
        output = val.decode()
    except (
        UnicodeDecodeError,
        AttributeError,
    ):
        pass

    return output
