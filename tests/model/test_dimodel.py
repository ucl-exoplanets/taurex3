import pytest
from hypothesis import given, settings, note
from hypothesis.strategies import floats, lists
from taurex.constants import get_constant
from taurex.util import conversion_factor
import numpy as np



@given(T_planet=floats(500, 1500), 
       planet_radius=floats(0.7*get_constant('R_earth', unit='m'), 1.5*get_constant('R_jup', unit='m')),
       system_distance=floats(1*conversion_factor('pc', 'm'), 100*conversion_factor('pc', 'm')))
def test_direct_image_final_flux(T_planet, planet_radius, system_distance):

    from taurex.model.directimage import compute_direct_image_final_flux
    from taurex.util.emission import black_body

    wlgrid = np.linspace(0.3, 20, 1000)
    f_planet = black_body(wlgrid, T_planet)
    val = f_planet*(planet_radius**2)/(system_distance**2)

    assert np.all(compute_direct_image_final_flux(f_planet, planet_radius, system_distance) == val)