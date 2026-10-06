"""Compatibility between YAML and legacy parameter files."""

from pathlib import Path

import configobj
import pytest
import yaml

from taurex.parameter import ParameterParser


EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "parfiles"


@pytest.mark.parametrize("source", sorted(EXAMPLES.glob("*.par")), ids=lambda p: p.name)
def test_yaml_matches_legacy_examples(source, tmp_path):
    """All example configurations retain their values and nesting in YAML."""
    target = tmp_path / "input.yaml"
    target.write_text(yaml.safe_dump(configobj.ConfigObj(str(source)).dict()))
    legacy, modern = ParameterParser(), ParameterParser()
    legacy.read(source)
    modern.read(target)
    assert modern._raw_config.dict() == legacy._raw_config.dict()


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".YAML"])
def test_yaml_values_and_factories(tmp_path, suffix):
    """Native YAML syntax supports existing profiles, lists and fitting keys."""
    target = tmp_path / ("input" + suffix)
    target.write_text(
        """
Temperature:
  profile_type: isothermal
  T: 1200
Chemistry:
  chemistry_type: taurex
  fill_gases: [H2, He]
  ratio: 0.17
  NO:
    gas_type: constant
    mix_ratio: 1e-4
Model:
  model_type: transmission
  Absorption: {}
Fitting:
  T:fit: true
  T:bounds: [1000, 1500]
  T:prior: 'Uniform(bounds=(1000, 1500))'
Global:
  xsec_in_memory: false
"""
    )
    parser = ParameterParser()
    parser.read(target)
    assert parser.generate_temperature_profile().isoTemperature == 1200
    assert parser.generate_temperature_profile().isoTemperature == 1200
    chemistry = parser.generate_chemistry_profile()
    assert "NO" in chemistry.gases
    config = parser._raw_config.dict()
    assert config["Chemistry"]["NO"]["mix_ratio"] == 1e-4
    assert config["Model"]["Absorption"] == {}
    assert config["Global"]["xsec_in_memory"] is False
    fitting = parser.generate_fitting_parameters()["T"]
    assert fitting["fit"] is True
    assert fitting["bounds"] == [1000, 1500]
    assert fitting["prior"] is not None


@pytest.mark.parametrize(
    "content, message",
    [
        ("", "mapping of sections"),
        ("- Model", "mapping of sections"),
        ("Model:", "sections must be mappings"),
        ("Temperature: {T: 1, T: 2}", "Duplicate YAML"),
        ("Model: {1: value, !!int 2: other}", "keys must be strings"),
        ("Model: {items: [{type: transit}]}", "list of scalars"),
        ("Model: &model {child: *model}", "Recursive YAML"),
    ],
)
def test_yaml_invalid_structure(tmp_path, content, message):
    """Unsupported structures fail early rather than silently changing meaning."""
    target = tmp_path / "invalid.yaml"
    target.write_text(content)
    with pytest.raises(ValueError, match=message):
        ParameterParser().read(target)


def test_yaml_rejects_python_objects(tmp_path):
    """YAML must not instantiate Python objects."""
    target = tmp_path / "unsafe.yaml"
    target.write_text("Model: !!python/object:builtins.object {}")
    with pytest.raises(yaml.constructor.ConstructorError):
        ParameterParser().read(target)


def test_yaml_syntax_error(tmp_path):
    """Malformed YAML retains the loader's source location in its error."""
    target = tmp_path / "invalid.yml"
    target.write_text("Model: [")
    with pytest.raises(yaml.YAMLError):
        ParameterParser().read(target)


def test_legacy_arbitrary_suffix(tmp_path):
    """Legacy files still work without a .par extension."""
    target = tmp_path / "input.txt"
    target.write_text("[Temperature]\nprofile_type = isothermal\nT = 900\n")
    parser = ParameterParser()
    parser.read(target)
    assert parser.generate_temperature_profile().isoTemperature == 900


def test_mixed_regional_files(tmp_path, monkeypatch):
    """Multimodel factories accept YAML and PAR regional files together."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "east.yaml").write_text(
        """
Temperature:
  profile_type: isothermal
  T: 800
Model:
  Rayleigh: {}
"""
    )
    (tmp_path / "west.par").write_text(
        "[Temperature]\nprofile_type = isothermal\nT = 1200\n"
        "[Model]\n[[Absorption]]\n"
    )
    (tmp_path / "main.yaml").write_text(
        """
Chemistry:
  chemistry_type: taurex
Model:
  model_type: multi_transit
  parfiles: [east.yaml, west.par]
  fractions: [0.5, 0.5]
"""
    )
    parser = ParameterParser()
    parser.read("main.yaml")
    model = parser.generate_model()
    regions = model.setup_keywords()
    assert [t.isoTemperature for t in regions["temperature_profiles"]] == [800, 1200]
    assert regions["chemistry"][0] is regions["chemistry"][1]
    assert [type(c[0]).__name__ for c in regions["contributions"]] == [
        "RayleighContribution",
        "AbsorptionContribution",
    ]


def test_quickstart_yaml_example():
    """The shipped quickstart examples describe the same model."""
    legacy, modern = ParameterParser(), ParameterParser()
    legacy.read(EXAMPLES / "quickstart.par")
    modern.read(EXAMPLES / "quickstart.yaml")
    assert legacy._raw_config.dict() == modern._raw_config.dict()


def test_inline_regions_match_files(tmp_path):
    """Inline regions preserve sharing, overrides and contribution replacement."""
    config = {
        "Planet": {"planet_type": "simple"},
        "Star": {"star_type": "blackbody"},
        "Temperature": {"profile_type": "isothermal", "T": 1100},
        "Pressure": {"profile_type": "simple", "nlayers": 15},
        "Chemistry": {"chemistry_type": "taurex", "ratio": 0.3},
        "Model": {
            "model_type": "multi_transit",
            "fractions": [0.5, 0.5],
            "Absorption": {},
            "regions": {
                "m1": {"Model": {"Rayleigh": {}}},
                "m2": {
                    "Temperature": {"profile_type": "isothermal", "T": 800},
                    "Pressure": {"profile_type": "simple", "nlayers": 20},
                    "Chemistry": {"chemistry_type": "taurex"},
                    "Model": {"Rayleigh": {}},
                },
            },
        },
    }
    target = tmp_path / "inline.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False))
    parser = ParameterParser()
    parser.read(target)
    model = parser.generate_model()
    inline = model.setup_keywords()
    assert inline["temperature_profiles"][0] is model._default_temperature
    assert inline["chemistry"][0] is model._default_chemistry
    assert inline["pressure_profile"][0] is model._default_pressure
    assert inline["chemistry"][1] is not model._default_chemistry
    assert inline["planet"] is model._planet
    assert inline["star"] is model._star
    assert inline["nlayers"] == [15, 20]
    assert [p.isoTemperature for p in inline["temperature_profiles"]] == [1100, 800]
    left, right = inline["contributions"]
    assert len(left) == len(right) == 1
    assert type(left[0]).__name__ == "RayleighContribution"
    assert type(right[0]) is type(left[0])
    assert left[0] is not right[0]

    paths = []
    for name, region in config["Model"].pop("regions").items():
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(region))
        paths.append(str(path))
    config["Model"]["parfiles"] = paths
    parser.read_dict(config)
    legacy = parser.generate_model().setup_keywords()
    assert legacy["nlayers"] == inline["nlayers"]
    assert [p.isoTemperature for p in legacy["temperature_profiles"]] == [1100, 800]
    assert [type(c[0]) for c in legacy["contributions"]] == [
        type(left[0]),
        type(right[0]),
    ]
    # An explicit regional chemistry uses constructor defaults, not a field merge.
    default = ParameterParser()
    default.read_dict({"Chemistry": {"chemistry_type": "taurex"}})
    assert (
        inline["chemistry"][1]._fill_ratio
        == default.generate_chemistry_profile()._fill_ratio
    )
    assert inline["chemistry"][1]._fill_ratio != inline["chemistry"][0]._fill_ratio


def test_inline_region_fallback():
    """Empty regions retain the legacy shared profiles and contribution fallback."""
    parser = ParameterParser()
    parser.read_dict(
        {
            "Chemistry": {"chemistry_type": "taurex"},
            "Temperature": {"profile_type": "isothermal", "T": 900},
            "Model": {
                "model_type": "multi_transit",
                "Rayleigh": {},
                "regions": {"m1": {}, "m2": {"Model": {}}},
            },
        }
    )
    model = parser.generate_model()
    regions = model.setup_keywords()
    for key in ["temperature_profiles", "chemistry", "pressure_profile"]:
        assert regions[key][0] is regions[key][1]
    assert regions["contributions"][0] is not regions["contributions"][1]
    assert regions["contributions"][0][0] is model.contribution_list[0]
    assert regions["contributions"][1][0] is model.contribution_list[0]


@pytest.mark.parametrize(
    "regions, message",
    [
        ({}, "non-empty mapping"),
        ([], "non-empty mapping"),
        ("m1", "non-empty mapping"),
        ({"m2": {}, "m1": {}}, "keys must be"),
        ({"m1": {}, "m3": {}}, "keys must be"),
        ({"east": {}}, "keys must be"),
        ({"m1": "bad"}, "mapping of sections"),
        ({"m1": {"Temperature": 800}}, "must be a mapping"),
        ({"m1": {"Planet": {}}}, "Unsupported regional section"),
        ({"m1": {"Star": {}}}, "Unsupported regional section"),
        ({"m1": {"Fitting": {}}}, "Unsupported regional section"),
        ({"m1": {"Global": {}}}, "Unsupported regional section"),
        ({"m1": {"Model": {"model_type": "transmission"}}}, "only contribution"),
        ({"m1": {"Model": {"regions": {}}}}, "only contribution"),
        ({"m1": {"Model": {"parfiles": []}}}, "only contribution"),
        ({"m1": {"Model": {"nlayers": 20}}}, "only contribution"),
    ],
)
def test_invalid_inline_regions(regions, message):
    """Invalid regional structure is rejected before model construction."""
    from taurex.model import MultiParameterTransitModel

    with pytest.raises(ValueError, match=message):
        MultiParameterTransitModel(regions=regions)


@pytest.mark.parametrize(
    "model_type", ["transmission", "multi_eclipse", "multi_directimage"]
)
def test_inline_regions_unsupported_model(model_type):
    """The reserved regions setting cannot be mistaken for a contribution."""
    parser = ParameterParser()
    parser.read_dict(
        {
            "Chemistry": {"chemistry_type": "taurex"},
            "Model": {"model_type": model_type, "regions": {"m1": {}}},
        }
    )
    with pytest.raises(ValueError, match="only supported for multi_transit"):
        parser.generate_model()


def test_inline_regions_conflict():
    """Even an empty parfiles setting conflicts with inline regions."""
    parser = ParameterParser()
    parser.read_dict(
        {
            "Chemistry": {"chemistry_type": "taurex"},
            "Model": {
                "model_type": "multi_transit",
                "parfiles": [],
                "regions": {"m1": {}},
            },
        }
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        parser.generate_model()


def test_inline_unknown_contribution():
    """A misspelled contribution must not silently select the fallback list."""
    parser = ParameterParser()
    parser.read_dict(
        {
            "Chemistry": {"chemistry_type": "taurex"},
            "Model": {
                "model_type": "multi_transit",
                "Rayleigh": {},
                "regions": {"m1": {"Model": {"UnknownContribution": {}}}},
            },
        }
    )
    with pytest.raises(KeyError, match="UnknownContribution"):
        parser.generate_model().setup_keywords()
