"""Render the legacy and working-tree depth plots with identical synthetic data.

Run from the repository root with PYTHONPATH=src. The JSON report records
observed differences; this diagnostic does not assert visual equivalence.
Only use trusted baseline revisions: their plotting method is executed.
"""

import argparse
import ast
import json
import subprocess  # noqa: S404 - reads a trusted local Git revision without a shell
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from taurex.plot.plotter import Plotter


matplotlib.use("Agg")


def legacy_method(revision):
    """Load only the original plotting method from a fixed Git revision."""
    source = subprocess.check_output(  # noqa: S603,S607 - fixed git command, no shell
        ["git", "show", f"{revision}:src/taurex/plot/plotter.py"], text=True
    )
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Plotter"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_plot_tau"
    )
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"plt": plt, "np": np}
    # Execute only the plotting method extracted from the trusted baseline.
    exec(compile(module, f"{revision}:_plot_tau", "exec"), namespace)  # noqa: S102
    return namespace["_plot_tau"]


def render(method, tau, pressure, wavelength, destination):
    """Save the rendering and capture data and axes for comparison."""
    plt.close("all")
    with plt.rc_context({"figure.figsize": (10, 5), "figure.dpi": 100}):
        method(None, tau, pressure, wavelength)
        fig = plt.gcf()
        fig.canvas.draw()
        fig.savefig(destination)
        main, _, profile = fig.axes
        result = {
            "pixels": np.asarray(fig.canvas.buffer_rgba()).copy(),
            "heatmap": np.asarray(
                main.images[0].get_array()
                if main.images
                else main.collections[0].get_array()
            ),
            "profile_x": profile.lines[0].get_xdata(),
            "profile_y": profile.lines[0].get_ydata(),
            "axes": {
                "heatmap_xlim": list(main.get_xlim()),
                "heatmap_ylim": list(main.get_ylim()),
                "heatmap_scales": [main.get_xscale(), main.get_yscale()],
                "profile_ylim": list(profile.get_ylim()),
            },
        }
        plt.close(fig)
        return result


def main():
    """Compare log-spaced and irregular grids against the legacy method."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", default="b0768b46903559798f9d69682e48e259e8f2575f"
    )
    parser.add_argument("--output", type=Path, default=Path("scratch/plot-comparison"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    old_method = legacy_method(args.baseline)
    report = {"baseline": args.baseline, "cases": {}}
    for name, pressure, wavelength in [
        ("log_grid", np.geomspace(1, 1e5, 12), np.geomspace(12, 0.5, 24)),
        ("descending_pressure", np.geomspace(1e5, 1, 12), np.geomspace(12, 0.5, 24)),
        (
            "irregular_grid",
            np.geomspace(3, 3e4, 12),
            np.array([12, 11, 10, 4, 2, 1, 0.7, 0.5]),
        ),
    ]:
        # An asymmetric feature exposes coordinate reversal and displacement.
        layer = np.linspace(0, 1, pressure.size)[:, None]
        spectrum = np.linspace(0, 1, wavelength.size)[None, :]
        tau = np.exp(-(((layer - 0.25 - 0.4 * spectrum) / 0.16) ** 2))
        tau *= 0.3 + spectrum
        old = render(
            old_method, tau, pressure, wavelength, args.output / f"{name}_old.png"
        )
        new = render(
            Plotter._plot_tau,
            tau,
            pressure,
            wavelength,
            args.output / f"{name}_new.png",
        )
        report["cases"][name] = {
            "heatmap_values_equal": bool(
                np.array_equal(old["heatmap"], new["heatmap"])
            ),
            "profile_values_equal": bool(
                np.allclose(old["profile_x"], new["profile_x"])
            ),
            "profile_pressures_equal": bool(
                np.allclose(old["profile_y"], new["profile_y"])
            ),
            "pixels_equal": bool(np.array_equal(old["pixels"], new["pixels"])),
            "changed_pixel_fraction": float(
                np.mean(np.any(old["pixels"] != new["pixels"], axis=2))
            ),
            "old_axes": old["axes"],
            "new_axes": new["axes"],
        }
    output = json.dumps(report, indent=2)
    (args.output / "report.json").write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
