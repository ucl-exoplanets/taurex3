"""Sphinx configuration."""
project = "TauREx3"
author = "Ahmed Faris Al-Refaie"
copyright = "2024, Ahmed Faris Al-Refaie"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
