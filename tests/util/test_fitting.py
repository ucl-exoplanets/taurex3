"""Test fitting utilities."""

import pytest


def test_priors_validate():
    """Test priors validate."""
    from taurex.util.fitting import MalformedPriorInputError
    from taurex.util.fitting import validate_priors

    validate_priors("(hello)")
    validate_priors("((hello),(there))")

    with pytest.raises(MalformedPriorInputError):
        validate_priors("(hello")

    with pytest.raises(MalformedPriorInputError):
        validate_priors("hello)")

    with pytest.raises(MalformedPriorInputError):
        validate_priors("((hello),(there)")

    with pytest.raises(MalformedPriorInputError):
        validate_priors("hello,there")


def test_parse_priors():
    """Test parse priors."""
    from taurex.util.fitting import parse_priors

    name, args = parse_priors(
        "HELLOWWORLD(var_a='this', "
        "var_b=[10, 20, 30, 40],"
        "var_c=1e34, var_d=False)"
    )

    assert name == "HELLOWWORLD"
    expected = {
        a: b
        for a, b in zip(
            ["var_a", "var_b", "var_c", "var_d"],
            ["this", [10, 20, 30, 40], 1e34, False],
            strict=False,
        )
    }
    assert args == expected
