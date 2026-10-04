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
    target.write_text("""
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
""")
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
    (tmp_path / "east.yaml").write_text("""
Temperature:
  profile_type: isothermal
  T: 800
Model:
  Rayleigh: {}
""")
    (tmp_path / "west.par").write_text(
        "[Temperature]\nprofile_type = isothermal\nT = 1200\n"
        "[Model]\n[[Absorption]]\n"
    )
    (tmp_path / "main.yaml").write_text("""
Chemistry:
  chemistry_type: taurex
Model:
  model_type: multi_transit
  parfiles: [east.yaml, west.par]
  fractions: [0.5, 0.5]
""")
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
