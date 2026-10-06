"""Environment expansion in file and inline parameter configurations."""

import pytest
import yaml

from taurex.parameter import ParameterParser


@pytest.mark.parametrize("suffix", ["yaml", "par"])
def test_environment_values(tmp_path, monkeypatch, suffix):
    """Paths and lists expand before the existing scalar conversion."""
    monkeypatch.setenv("TAUREX_TEST_DATA", "/data/with spaces")
    monkeypatch.setenv("TAUREX_TEST_TEMP", "1200")
    monkeypatch.setenv("TAUREX_TEST_MEMORY", "false")
    monkeypatch.setenv("TAUREX_TEST_BOUND", "1500")
    config = {
        "Global": {
            "xsec_path": "$TAUREX_TEST_DATA/xsec",
            "cia_path": "${TAUREX_TEST_DATA}/cia",
            "xsec_in_memory": "$TAUREX_TEST_MEMORY",
        },
        "Temperature": {"profile_type": "isothermal", "T": "$TAUREX_TEST_TEMP"},
        "Model": {
            "parfiles": ["${TAUREX_TEST_DATA}/east.yaml", "$TAUREX_TEST_DATA/west.par"]
        },
        "Fitting": {"T:bounds": ["1000", "$TAUREX_TEST_BOUND"]},
    }
    target = tmp_path / f"input.{suffix}"
    if suffix == "yaml":
        target.write_text(yaml.safe_dump(config))
    else:
        import configobj

        legacy = configobj.ConfigObj(config)
        legacy.filename = str(target)
        legacy.write()
    parser = ParameterParser()
    parser.read(target)
    result = parser._raw_config.dict()
    assert result["Global"]["xsec_path"] == "/data/with spaces/xsec"
    assert result["Global"]["cia_path"] == "/data/with spaces/cia"
    assert result["Global"]["xsec_in_memory"] is False
    assert parser.generate_temperature_profile().isoTemperature == 1200
    assert result["Model"]["parfiles"] == [
        "/data/with spaces/east.yaml",
        "/data/with spaces/west.par",
    ]
    assert result["Fitting"]["T:bounds"] == [1000, 1500]


@pytest.mark.parametrize(
    "value",
    [
        "$TAUREX_TEST_MISSING",
        "${TAUREX_TEST_MISSING}/x",
        ["ok", "$TAUREX_TEST_MISSING"],
    ],
)
def test_missing_environment_variable(monkeypatch, value):
    """Missing references report the variable and parameter instead of a bad path."""
    monkeypatch.delenv("TAUREX_TEST_MISSING", raising=False)
    with pytest.raises(ValueError, match="TAUREX_TEST_MISSING.*xsec_path"):
        ParameterParser().read_dict({"Global": {"xsec_path": value}})


def test_environment_is_single_pass(monkeypatch):
    """Escapes, empty variables and literal shell text have predictable behavior."""
    monkeypatch.setenv("TAUREX_TEST_LITERAL", "$TAUREX_TEST_MISSING")
    monkeypatch.setenv("TAUREX_TEST_EMPTY", "")
    monkeypatch.delenv("TAUREX_TEST_MISSING", raising=False)
    parser = ParameterParser()
    parser.read_dict(
        {
            "Global": {
                "literal": "$$TAUREX_TEST_MISSING/${TAUREX_TEST_EMPTY}",
                "replacement": "$TAUREX_TEST_LITERAL",
                "empty": "$TAUREX_TEST_EMPTY",
                "shell": "$(echo untouched)/${UNSET:-fallback}",
                "$TAUREX_TEST_MISSING": "key is unchanged",
            }
        }
    )
    values = parser._raw_config.dict()["Global"]
    assert values["literal"] == "$TAUREX_TEST_MISSING/"
    assert values["replacement"] == "$TAUREX_TEST_MISSING"
    assert values["empty"] == ""
    assert values["shell"] == "$(echo untouched)/${UNSET:-fallback}"
    assert values["$TAUREX_TEST_MISSING"] == "key is unchanged"


def test_inline_environment_expanded_once(tmp_path, monkeypatch):
    """Inline regions expand on input and never reinterpret replacement text."""
    from taurex.model import MultiParameterTransitModel

    monkeypatch.setenv("TAUREX_TEST_TEMP", "800")
    monkeypatch.delenv("TAUREX_TEST_LITERAL", raising=False)
    target = tmp_path / "inline.yaml"
    target.write_text(
        """
Chemistry:
  chemistry_type: taurex
Model:
  model_type: multi_transit
  regions:
    m1:
      Temperature:
        profile_type: isothermal
        T: $TAUREX_TEST_TEMP
      Model:
        CIA:
          cia_pairs: ["$$TAUREX_TEST_LITERAL", H2-He]
"""
    )
    parser = ParameterParser()
    parser.read(target)
    seen = []
    original = MultiParameterTransitModel._read_region

    def inspect_region(self, regional_parser):
        seen.append(regional_parser._raw_config.dict())
        return original(self, regional_parser)

    monkeypatch.setattr(MultiParameterTransitModel, "_read_region", inspect_region)
    regions = parser.generate_model().setup_keywords()
    assert regions["temperature_profiles"][0].isoTemperature == 800
    assert seen[0]["Model"]["CIA"]["cia_pairs"][0] == "$TAUREX_TEST_LITERAL"
