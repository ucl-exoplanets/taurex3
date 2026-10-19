"""Alias for chemistry profiles."""

from taurex.data.profiles.chemistry import AutoChemistry
from taurex.data.profiles.chemistry import Chemistry
from taurex.data.profiles.chemistry import ChemistryFile
from taurex.data.profiles.chemistry import ConstantGas
from taurex.data.profiles.chemistry import CustomGas
from taurex.data.profiles.chemistry import Gas
from taurex.data.profiles.chemistry import PowerGas
from taurex.data.profiles.chemistry import TaurexChemistry
from taurex.data.profiles.chemistry import TwoLayerGas
from taurex.data.profiles.chemistry import TwoPointGas
from taurex.data.profiles.chemistry.gas.arraygas import ArrayGas


__all__ = [
    "Chemistry",
    "Gas",
    "ConstantGas",
    "TwoLayerGas",
    "TwoPointGas",
    "PowerGas",
    "CustomGas",
    "TaurexChemistry",
    "AutoChemistry",
    "ChemistryFile",
    "ArrayGas",
]
