from __future__ import annotations

from pathlib import Path
import sys
from urllib.request import urlretrieve

EXOMOL_URLS = {
    "H2O": "https://exomol.com/db/H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL__R15000_0.3-50mu.xsec.TauREx.h5",
    "CO2": "https://exomol.com/db/CO2/12C-16O2/UCL-4000/12C-16O2__UCL-4000.R15000_0.3-50mu.xsec.TauREx.h5",
    "CH4": "https://exomol.com/db/CH4/12C-1H4/MM/12C-1H4__MM.R15000_0.3-50mu.xsec.TauREx.h5",
    "NH3": "https://exomol.com/db/NH3/14N-1H3/CoYuTe/14N-1H3__CoYuTe.R15000_0.3-50mu.xsec.TauREx.h5",
}

CIA_URLS = {
    "H2-H2_eq_2018.cia": "https://hitran.org/data/CIA/alternate/H2-H2_eq_2018.cia",
    "H2-He_eq_2011.cia": "https://hitran.org/data/CIA/alternate/H2-He_eq_2011.cia",
}


def find_project_root(start_path: Path | None = None) -> Path:
    current = start_path or Path.cwd()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "taurex").exists():
            return candidate
    raise FileNotFoundError("Could not locate the TauREx project root.")


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
TMP_DIR = PROJECT_ROOT / "examples" / "tmp"
XSEC_DIR = TMP_DIR / "xsec"
CIA_DIR = TMP_DIR / "cia"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

XSEC_DIR.mkdir(parents=True, exist_ok=True)
CIA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    urlretrieve(url, destination)
    return destination


def ensure_opacity_data(download: bool = False) -> None:
    required_xsec = [XSEC_DIR / url.rsplit("/", 1)[-1] for url in EXOMOL_URLS.values()]
    required_cia = [CIA_DIR / filename for filename in CIA_URLS]

    if download:
        for url in EXOMOL_URLS.values():
            download_file(url, XSEC_DIR / url.rsplit("/", 1)[-1])
        for filename, url in CIA_URLS.items():
            download_file(url, CIA_DIR / filename)

    missing = [path for path in [*required_xsec, *required_cia] if not path.exists()]
    if missing:
        missing_list = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing opacity data. Run ensure_opacity_data(download=True) first:\n" + missing_list
        )

    from taurex.cache import CIACache, OpacityCache

    OpacityCache().set_opacity_path(str(XSEC_DIR))
    CIACache().set_cia_path(str(CIA_DIR))


def build_base_components(download: bool = False, nlayers: int = 100) -> dict[str, object]:
    ensure_opacity_data(download=download)

    from taurex.chemistry import ConstantGas, TaurexChemistry
    from taurex.planet import Planet
    from taurex.pressure import SimplePressureProfile
    from taurex.stellar import BlackbodyStar
    from taurex.temperature import Isothermal

    iso_t = Isothermal(T=2000.0)
    press = SimplePressureProfile(nlayers=nlayers, atm_min_pressure=1e-5, atm_max_pressure=1e5)
    chemistry = TaurexChemistry(fill_gases=["H2", "He"], ratio=0.1756)
    chemistry.addGas(ConstantGas(molecule_name="H2O", mix_ratio=1e-3))
    chemistry.addGas(ConstantGas(molecule_name="CH4", mix_ratio=1e-4))
    chemistry.addGas(ConstantGas(molecule_name="NH3", mix_ratio=1e-4))
    chemistry.addGas(ConstantGas(molecule_name="CO2", mix_ratio=1e-4))
    planet = Planet(planet_mass=0.74, planet_radius=1.38)
    star = BlackbodyStar(temperature=6117, radius=1.16)

    press.compute_pressure_profile()
    iso_t.initialize_profile(
        planet=planet,
        nlayers=press.nLayers,
        pressure_profile=press.profile,
    )
    chemistry.set_star_planet(star=star, planet=planet)
    chemistry.initialize_chemistry(
        nlayers=press.nLayers,
        temperature_profile=iso_t.profile,
        pressure_profile=press.profile,
    )

    return {
        "iso_t": iso_t,
        "press": press,
        "chemistry": chemistry,
        "planet": planet,
        "star": star,
    }


def build_transmission_model(
    include_cia: bool = False,
    include_rayleigh: bool = False,
    clouds=None,
    download: bool = False,
    nlayers: int = 100,
) -> dict[str, object]:
    from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
    from taurex.model import TransmissionModel

    context = build_base_components(download=download, nlayers=nlayers)
    model = TransmissionModel(
        planet=context["planet"],
        temperature_profile=context["iso_t"],
        chemistry=context["chemistry"],
        star=context["star"],
        pressure_profile=context["press"],
    )
    model.add_contribution(AbsorptionContribution())
    if include_cia:
        model.add_contribution(CIAContribution(cia_pairs=["H2-H2", "H2-He"]))
    if include_rayleigh:
        model.add_contribution(RayleighContribution())
    if clouds is not None:
        model.add_contribution(clouds)
    model.build()
    return {**context, "tm": model}


def build_emission_model(
    temperature_profile=None,
    include_cia: bool = True,
    include_rayleigh: bool = True,
    download: bool = False,
    nlayers: int = 100,
) -> dict[str, object]:
    from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
    from taurex.model import EmissionModel

    context = build_base_components(download=download, nlayers=nlayers)
    selected_temperature = temperature_profile or context["iso_t"]
    selected_temperature.initialize_profile(
        planet=context["planet"],
        nlayers=context["press"].nLayers,
        pressure_profile=context["press"].profile,
    )
    context["chemistry"].initialize_chemistry(
        nlayers=context["press"].nLayers,
        temperature_profile=selected_temperature.profile,
        pressure_profile=context["press"].profile,
    )
    model = EmissionModel(
        planet=context["planet"],
        temperature_profile=selected_temperature,
        chemistry=context["chemistry"],
        star=context["star"],
        pressure_profile=context["press"],
    )
    model.add_contribution(AbsorptionContribution())
    if include_cia:
        model.add_contribution(CIAContribution(cia_pairs=["H2-H2", "H2-He"]))
    if include_rayleigh:
        model.add_contribution(RayleighContribution())
    model.build()
    return {**context, "em": model}
