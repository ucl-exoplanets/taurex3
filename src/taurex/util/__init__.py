"""Common functions that are used and are quite helpful."""

from .util import bindown
from .util import calculate_weight
from .util import class_for_name
from .util import class_from_keyword
from .util import clip_native_to_wngrid
from .util import compute_bin_edges
from .util import compute_dz
from .util import conversion_factor
from .util import create_grid_res
from .util import decode_string_array
from .util import ensure_string_utf8
from .util import find_closest_pair
from .util import get_molecular_weight
from .util import has_duplicates
from .util import loadtxt2d
from .util import mass
from .util import merge_elements
from .util import molecule_texlabel
from .util import movingaverage
from .util import quantile_corner
from .util import random_int_iter
from .util import read_error_into_dict
from .util import read_error_line
from .util import read_table
from .util import recursively_save_dict_contents_to_output
from .util import sanitize_molecule_string
from .util import split_molecule_elements
from .util import store_thing
from .util import tokenize_molecule
from .util import weighted_avg_and_std
from .util import wnwidth_to_wlwidth


__all__ = [
    "mass",
    "calculate_weight",
    "split_molecule_elements",
    "tokenize_molecule",
    "merge_elements",
    "sanitize_molecule_string",
    "get_molecular_weight",
    "molecule_texlabel",
    "bindown",
    "movingaverage",
    "quantile_corner",
    "loadtxt2d",
    "read_error_line",
    "read_error_into_dict",
    "read_table",
    "decode_string_array",
    "recursively_save_dict_contents_to_output",
    "store_thing",
    "weighted_avg_and_std",
    "random_int_iter",
    "compute_bin_edges",
    "clip_native_to_wngrid",
    "wnwidth_to_wlwidth",
    "class_from_keyword",
    "class_for_name",
    "create_grid_res",
    "conversion_factor",
    "compute_dz",
    "has_duplicates",
    "find_closest_pair",
    "ensure_string_utf8",
]
