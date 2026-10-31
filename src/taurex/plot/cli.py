"""Command-line adapter for combined HDF5 plots."""

from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

from .comparison import plot_gas_profile_files
from .comparison import plot_posterior_files
from .comparison import plot_spectrum_files
from .comparison import plot_temperature_pressure_files


def comparison_mode(args, parser):
    """Validate comparison arguments and render PDFs; return whether handled.

    A single input uses the legacy Plotter. Comparison-only arguments are
    rejected there rather than silently ignored. All validation errors use
    argparse's standard error reporting and exit status.
    """
    if len(args.input_file) == 1:
        if args.labels is not None or args.parameters is not None:
            parser.error("--labels and --parameters require multiple input files")
        return False
    _validate(args, parser)
    try:
        _render(args)
    except (OSError, KeyError, ValueError) as error:
        parser.error(f"cannot plot combined outputs: {error}")
    return True


def _validate(args, parser):
    if args.labels is not None and len(args.labels) != len(args.input_file):
        parser.error("--labels must supply one label per input file")
    if args.tau or args.cprofile or args.contrib or args.full_contrib:
        parser.error(
            "multiple inputs do not support tau, condensate or contribution plots"
        )
    if args.resolution is not None:
        parser.error("--resolution is not supported with multiple input files")
    if not any(
        (args.all, args.tpprofile, args.xprofile, args.spectrum, args.posterior)
    ):
        parser.error("select at least one comparison plot or --all")
    if args.parameters is not None and not (args.posterior or args.all):
        parser.error("--parameters requires --plot-posteriors or --all")
    if args.posterior or args.all:
        _require_retrievals(args.input_file, parser)


def _require_retrievals(filenames, parser):
    for filename in filenames:
        try:
            with h5py.File(filename, "r") as fd:
                if "Output/Solutions" not in fd:
                    parser.error("combined posteriors require retrieval files only")
        except OSError as error:
            parser.error(str(error))


def _render(args):
    output = Path(args.output_dir)
    prefix = args.prefix if args.prefix is not None else "output"
    count = max(len(args.input_file), 2)
    cmap = mpl.colormaps[args.cmap]
    colors = [cmap(index / (count - 1)) for index in range(count)]
    output.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"axes.prop_cycle": mpl.cycler(color=colors)}):
        for selected, function, suffix in (
            (args.tpprofile, plot_temperature_pressure_files, "tp_profile"),
            (args.xprofile, plot_gas_profile_files, "mixratio"),
            (args.spectrum, plot_spectrum_files, "spectrum"),
        ):
            if selected or args.all:
                fig, ax = plt.subplots()
                try:
                    function(ax, args.input_file, labels=args.labels)
                    ax.legend()
                    _save(fig, output, prefix, suffix, args.title)
                finally:
                    plt.close(fig)
        if args.posterior or args.all:
            fig = plot_posterior_files(
                args.input_file, labels=args.labels, parameters=args.parameters
            )
            try:
                _save(fig, output, prefix, "posteriors", args.title)
            finally:
                plt.close(fig)


def _save(fig, output, prefix, suffix, title):
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_{suffix}.pdf")
