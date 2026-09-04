"""Tests for unit validation decorator."""

import pytest
from astropy import constants as const
from astropy import units as u

from taurex.util.unit import validate_arg_units


# Helper functions for testing
def test_decorator_basic_conversion():
    """Test basic unit conversion on positional arguments."""

    @validate_arg_units({"distance": {"target_unit": u.m, "equivalencies": None}})
    def calculate_time(distance: float, speed: float = 10.0) -> float:
        return distance / speed

    # Test with km that should convert to m
    result = calculate_time(5.0 * u.km, 10.0)  # 5 km = 5000 m
    assert result == 500.0  # 5000 m / 10 m/s = 500 s

    # Test with already in target unit
    result = calculate_time(5000.0 * u.m, 10.0)
    assert result == 500.0

    # Check if target is assumed
    result = calculate_time(5000.0, 10.0)
    assert result == 500.0


def test_decorator_with_default_unit():
    """Test conversion using default unit when unitless value provided."""

    @validate_arg_units(
        {"distance": {"target_unit": u.m, "default_unit": u.km, "equivalencies": None}}
    )
    def calculate_time(distance: float, speed: float = 10.0) -> float:
        return distance / speed

    # Check if km
    result = calculate_time(5.0, 10.0)  # 5 km = 5000 m
    assert result == 500.0  # 5000 m / 10 m/s = 500 s

    # Test with already in target unit
    result = calculate_time(5000.0 * u.m, 10.0)
    assert result == 500.0


def test_decorator_with_equivalencies():
    """Test conversion with equivalencies (e.g., spectral)."""

    @validate_arg_units(
        {
            "frequency": {
                "target_unit": u.Hz,
                "default_unit": u.Hz,
                "equivalencies": u.spectral(),
            }
        }
    )
    def process_wavelength(frequency) -> float:
        return frequency

    # Test with wavelength that should convert to frequency using spectral equivalency
    result = process_wavelength(5000 * u.AA) << u.Hz
    # 5000 Å corresponds to ~5.996e14 Hz
    expected_freq = (const.c / (5000 * u.AA)).to(u.Hz)
    assert abs(result - expected_freq) < 1e-10 * expected_freq


def test_decorator_multiple_parameters():
    """Test decorator with multiple parameters."""

    @validate_arg_units(
        {
            "mass": {"target_unit": u.kg, "default_unit": u.g, "equivalencies": None},
            "velocity": {
                "target_unit": u.m / u.s,
                "default_unit": u.km / u.s,
                "equivalencies": None,
            },
        }
    )
    def calculate_kinetic_energy(mass: float, velocity: float) -> float:
        return 0.5 * mass * velocity**2

    # Test with mixed units
    result = calculate_kinetic_energy(1000.0 * u.g, 10.0 * u.km / u.s)
    assert result == 5e7


def test_decorator_kwargs():
    """Test decorator with keyword arguments."""

    @validate_arg_units(
        {"radius": {"target_unit": u.m, "default_unit": u.km, "equivalencies": None}}
    )
    def calculate_area(*, radius: float) -> float:
        return 4 * 3.14159 * radius**2

    # Test with keyword argument
    result = calculate_area(radius=2.0 * u.km)
    expected = 4 * 3.14159 * (2000.0) ** 2  # 2km = 2000m
    assert abs(result - expected) < 1e-10


def test_decorator_mixed_args_kwargs():
    """Test decorator with both positional and keyword arguments."""

    @validate_arg_units(
        {
            "length": {"target_unit": u.m, "default_unit": u.cm, "equivalencies": None},
            "width": {"target_unit": u.m, "default_unit": u.cm, "equivalencies": None},
        }
    )
    def calculate_rectangle_area(length: float, width: float) -> float:
        return length * width

    # Mix positional and keyword
    result = calculate_rectangle_area(100.0 * u.cm, width=200.0 * u.cm)
    # 100cm = 1m, 200cm = 2m, area = 2m²
    assert result == 2.0


def test_decorator_no_conversion_needed():
    """Test decorator when values already have correct units."""

    @validate_arg_units(
        {"time": {"target_unit": u.s, "default_unit": u.minute, "equivalencies": None}}
    )
    def double_time(time: float) -> float:
        return 2 * time

    # Already in seconds
    result = double_time(10.0 * u.s)
    assert result == 20.0


def test_decorator_with_string_units():
    """Test decorator with string unit specifications."""

    @validate_arg_units(
        {"distance": {"target_unit": "m", "default_unit": "km", "equivalencies": None}}
    )
    def scale_distance(distance: float, factor: float = 2.0) -> float:
        return distance * factor

    # Test with km that should convert to m
    result = scale_distance(5.0 * u.km)
    assert result == 10000.0


def test_decorator_with_none_equivalencies():
    """Test decorator with explicit None equivalencies."""

    @validate_arg_units(
        {"angle": {"target_unit": u.rad, "default_unit": u.deg, "equivalencies": None}}
    )
    def double_angle(angle: float) -> float:
        return 2 * angle

    result = double_angle(180.0 * u.deg)
    assert result == pytest.approx(2 * 3.14159)


def test_decorator_with_function_defaults():
    """Test decorator preserves function defaults."""

    @validate_arg_units(
        {"distance": {"target_unit": u.m, "default_unit": u.km, "equivalencies": None}}
    )
    def travel_time(distance: float, speed: float = 10.0) -> float:
        return distance / speed

    # Test with default speed
    result = travel_time(10.0 * u.km)
    assert result == 1000.0  # 10000m / 10 m/s = 1000s


# Edge cases and error handling
def test_decorator_parameter_not_in_definition():
    """Test decorator handles parameters not in unit_def."""

    @validate_arg_units(
        {"param1": {"target_unit": u.m, "default_unit": u.cm, "equivalencies": None}}
    )
    def test_function(param1: float, param2: float) -> float:
        return param1 + param2

    # param2 should pass through unchanged
    result = test_function(100.0 * u.cm, 5.0)
    assert result == 1.0 + 5.0


def test_decorator_without_units_passes_through():
    """Test decorator passes through values without units."""

    @validate_arg_units({})
    def identity(value: float) -> float:
        return value

    result = identity(42.0)
    assert result == 42.0


def test_decorator_partial_unit_definition():
    """Test decorator with partial unit definitions."""

    @validate_arg_units(
        {
            "temperature": {
                "target_unit": u.K,
                "default_unit": u.deg_C,
                "equivalencies": u.temperature(),
            }
            # "pressure" not defined - should pass through
        }
    )
    def thermo_calc(temperature: float, pressure: float = 1.0) -> float:
        return temperature * pressure

    result = thermo_calc(20.0 * u.deg_C, pressure=2.0)
    # 20°C = 293.15K, * 2 = 586.3
    assert result == 293.15 * 2.0


def test_decorator_complex_unit_definitions():
    """Test decorator with complex unit definitions."""

    @validate_arg_units(
        {
            "density": {
                "target_unit": u.kg / u.m**3,
                "default_unit": u.g / u.cm**3,
                "equivalencies": None,
            }
        }
    )
    def process_density(density: float) -> float:
        return density

    result = process_density(1.0 * u.g / u.cm**3)
    # 1 g/cm³ = 1000 kg/m³
    assert result == pytest.approx(1000.0)


# Class unit
def test_class_init():
    """Check to see if class methods also work."""
    expected = 120000.0

    class MyClass:

        @validate_arg_units(
            {
                "mass": {
                    "default_unit": u.kg,
                    "target_unit": u.g,
                },
                "velocity": {
                    "default_unit": "m/s",
                    "target_unit": "m/s",
                },
            }
        )
        def __init__(self, mass, velocity):
            self.momentum = mass * velocity

    instance = MyClass(10 << u.kg, 12 << u.m / u.s)

    assert instance.momentum == expected


def test_prefined_units():
    """Try TauREx predefined units."""
    from taurex.util.unit import DEFAULT_PRESSURE
    from taurex.util.unit import DEFAULT_SPECTRUM
    from taurex.util.unit import DEFAULT_TEMPERATURE

    @validate_arg_units(
        {
            "temperature": DEFAULT_TEMPERATURE,
            "pressure": DEFAULT_PRESSURE,
        }
    )
    def dens_function(temperature: float, pressure: float):
        return temperature / pressure

    assert pytest.approx(0.0127315) == dens_function(1000 << u.deg_C, 1 << u.bar)

    @validate_arg_units({"wngrid": DEFAULT_SPECTRUM})
    def check_spectrum(wngrid):
        return wngrid

    assert check_spectrum(10000 << u.um) == 1.0
