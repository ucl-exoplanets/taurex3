"""Plotting module for TauREx."""

import os

import h5py
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import taurex.plot.corner as corner
from taurex.util.util import decode_string_array


matplotlib.use("Agg")

# some global matplotlib vars
mpl.rcParams["axes.linewidth"] = 1  # set the value globally
mpl.rcParams["text.antialiased"] = True
mpl.rcParams["errorbar.capsize"] = 2

# rc('text', usetex=True) # use tex in plots
# rc('font', **{ 'family' : 'serif','serif':['Palatino'], 'size'   : 11})


class Plotter:
    """Plotting class for TauREx output."""

    phi = 1.618

    model_axis = {
        "TransmissionModel": "$(R_p/R_*)^2$",
        "EmissionModel": "$F_p/F_*$",
        "DirectImageModel": "$F_p$",
    }

    def __init__(
        self,
        filename,
        title=None,
        prefix=None,
        cmap="Paired",
        out_folder=".",
    ):
        """Initialize Plotter."""
        self.fd = h5py.File(filename, "r")
        self.title = title
        self.cmap = mpl.cm.get_cmap(cmap)
        self.prefix = prefix
        if self.prefix is None:
            self.prefix = "output"
        self.out_folder = out_folder

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

    def find_nearest(self, array, value):
        """Find nearest value in array."""
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    @property
    def num_solutions(self, fd_position="Output"):
        """Get number of solutions."""
        return len(
            [
                (int(k[8:]), v)
                for k, v in self.fd[fd_position]["Solutions"].items()
                if "solution" in k
            ]
        )

    def solution_iter(self, fd_position="Output"):
        """Iterate over solutions."""
        yield from [
            (int(k[8:]), v)
            for k, v in self.fd[fd_position]["Solutions"].items()
            if "solution" in k
        ]

    def forward_output(self):
        """Get forward output group."""
        return self.fd["Output"]

    def compute_ranges(self, mu=True, selected_fitparams=None):
        """Compute ranges."""
        solution_ranges = []

        mu_derived = None
        for _, sol in self.solution_iter():

            mu_derived = self.get_derived_parameters(sol)

            fitting_names = selected_fitparams or self.fittingNames

            fit_params = sol["fit_params"]
            param_list = []
            for fit_names in fitting_names:
                if fit_names not in self.fittingNames:
                    continue
                param_values = fit_params[fit_names]
                sigma_m = param_values["sigma_m"][()]
                sigma_p = param_values["sigma_p"][()]
                val = param_values["value"][()]

                param_list.append(
                    [
                        val,
                        val - 5.0 * sigma_m,
                        val + 5.0 * sigma_p,
                    ]
                )

            for d in mu_derived:
                sigma_m = d["sigma_m"][()]
                sigma_p = d["sigma_p"][()]
                val = d["value"][()]
                param_list.append(
                    [
                        val,
                        val - 5.0 * sigma_m,
                        val + 5.0 * sigma_p,
                    ]
                )

            solution_ranges.append(param_list)

        fitting_boundary_low = [
            self.fittingBoundaryLow[self.fittingNames.index(name)]
            for name in selected_fitparams
        ]
        fitting_boundary_high = [
            self.fittingBoundaryHigh[self.fittingNames.index(name)]
            for name in selected_fitparams
        ]
        if len(mu_derived) > 0:
            fitting_boundary_low = np.concatenate(
                (
                    fitting_boundary_low,
                    [-1e99] * len(mu_derived),
                )
            )
            fitting_boundary_high = np.concatenate(
                (
                    fitting_boundary_high,
                    [1e99] * len(mu_derived),
                )
            )

        range_all = np.array(solution_ranges)

        range_min = np.min(range_all[:, :, 1], axis=0)
        range_max = np.max(range_all[:, :, 2], axis=0)

        range_min = np.where(
            range_min < fitting_boundary_low,
            fitting_boundary_low,
            range_min,
        )
        range_max = np.where(
            range_max > fitting_boundary_high,
            fitting_boundary_high,
            range_max,
        )
        return list(zip(range_min, range_max, strict=True))

    @property
    def activeGases(self):
        """Get active gases."""
        return decode_string_array(
            self.fd["ModelParameters"]["Chemistry"]["active_gases"]
        )

    @property
    def condensates(self):
        """Get condensates."""
        return decode_string_array(
            self.fd["ModelParameters"]["Chemistry"]["condensates"]
        )

    @property
    def inactiveGases(self):
        """Get inactive gases."""
        return decode_string_array(
            self.fd["ModelParameters"]["Chemistry"]["inactive_gases"]
        )

    def plot_fit_xprofile(self):
        """Plot fitted mixing ratio profiles."""
        for solution_idx, solution_val in self.solution_iter():

            fig = plt.figure(figsize=(7, 7 / self.phi))
            ax = fig.add_subplot(111)
            num_moles = len(self.activeGases)

            profiles = solution_val["Profiles"]
            pressure_profile = profiles["pressure_profile"][:] / 1e5
            active_profile = profiles["active_mix_profile"][...]
            active_profile_std = profiles["active_mix_profile_std"][...]

            inactive_profile = profiles["inactive_mix_profile"][...]
            inactive_profile_std = profiles["inactive_mix_profile_std"][...]

            cols_mol = {}
            for mol_idx, mol_name in enumerate(self.activeGases):
                cols_mol[mol_name] = self.cmap(mol_idx / num_moles)

                prof = active_profile[mol_idx]
                prof_std = active_profile_std[mol_idx]

                plt.plot(
                    prof,
                    pressure_profile,
                    color=cols_mol[mol_name],
                    label=mol_name,
                )

                plt.fill_betweenx(
                    pressure_profile,
                    prof + prof_std,
                    prof,
                    color=self.cmap(mol_idx / num_moles),
                    alpha=0.5,
                )
                plt.fill_betweenx(
                    pressure_profile,
                    prof,
                    np.power(
                        10,
                        (np.log10(prof) - (np.log10(prof + prof_std) - np.log10(prof))),
                    ),
                    color=self.cmap(mol_idx / num_moles),
                    alpha=0.5,
                )

            plt.yscale("log")
            plt.gca().invert_yaxis()
            plt.xscale("log")
            plt.xlim(1e-12, 3)
            plt.xlabel("Mixing ratio")
            plt.ylabel("Pressure (bar)")
            plt.tight_layout()
            box = ax.get_position()
            ax.set_position(
                [
                    box.x0,
                    box.y0,
                    box.width * 0.8,
                    box.height,
                ]
            )
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                ncol=1,
                prop={"size": 11},
                frameon=False,
            )
            if self.title:
                plt.title(self.title + " - Active", fontsize=14)
            plt.savefig(
                os.path.join(
                    self.out_folder,
                    "%s_fit_active_mixratio_sol%i.pdf" % (self.prefix, solution_idx),
                )
            )
            plt.close("all")

        for solution_idx, solution_val in self.solution_iter():

            fig = plt.figure(figsize=(7, 7 / self.phi))
            ax = fig.add_subplot(111)
            num_moles = len(self.inactiveGases)

            profiles = solution_val["Profiles"]
            pressure_profile = profiles["pressure_profile"][:] / 1e5
            active_profile = profiles["active_mix_profile"][...]
            active_profile_std = profiles["active_mix_profile_std"][...]

            inactive_profile = profiles["inactive_mix_profile"][...]
            inactive_profile_std = profiles["inactive_mix_profile_std"][...]

            cols_mol = {}

            for mol_idx, mol_name in enumerate(self.inactiveGases):
                inactive_idx = len(self.activeGases) + mol_idx
                cols_mol[mol_name] = self.cmap(inactive_idx / num_moles)

                prof = inactive_profile[mol_idx]
                prof_std = inactive_profile_std[mol_idx]

                plt.plot(
                    prof,
                    pressure_profile,
                    color=cols_mol[mol_name],
                    label=mol_name,
                )

                plt.fill_betweenx(
                    pressure_profile,
                    prof + prof_std,
                    prof,
                    color=self.cmap(inactive_idx / num_moles),
                    alpha=0.5,
                )
                plt.fill_betweenx(
                    pressure_profile,
                    prof,
                    np.power(
                        10,
                        (np.log10(prof) - (np.log10(prof + prof_std) - np.log10(prof))),
                    ),
                    color=self.cmap(inactive_idx / num_moles),
                    alpha=0.5,
                )

            plt.yscale("log")
            plt.gca().invert_yaxis()
            plt.xscale("log")
            plt.xlim(1e-12, 3)
            plt.xlabel("Mixing ratio")
            plt.ylabel("Pressure (bar)")
            plt.tight_layout()
            box = ax.get_position()
            ax.set_position(
                [
                    box.x0,
                    box.y0,
                    box.width * 0.8,
                    box.height,
                ]
            )
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                ncol=1,
                prop={"size": 11},
                frameon=False,
            )
            if self.title:
                plt.title(self.title + "- Inactive", fontsize=14)
            plt.savefig(
                os.path.join(
                    self.out_folder,
                    "%s_fit_inactive_mixratio_sol%i.pdf" % (self.prefix, solution_idx),
                )
            )
            plt.close("all")

    def plot_forward_xprofile(self):
        """Plot forward mixing ratio profiles."""
        solution_val = self.forward_output()

        profiles = solution_val["Profiles"]
        pressure_profile = profiles["pressure_profile"][:] / 1e5
        active_profile = profiles["active_mix_profile"][...]

        inactive_profile = profiles["inactive_mix_profile"][...]

        cols_mol = {}

        fig = plt.figure(figsize=(7, 7 / self.phi))
        ax = fig.add_subplot(111)
        num_moles = len(self.activeGases)

        for mol_idx, mol_name in enumerate(self.activeGases):
            cols_mol[mol_name] = self.cmap(mol_idx / num_moles)

            prof = active_profile[mol_idx]

            plt.plot(
                prof,
                pressure_profile,
                color=cols_mol[mol_name],
                label=mol_name,
            )

        plt.yscale("log")
        plt.gca().invert_yaxis()
        plt.xscale("log")
        plt.xlim(1e-12, 3)
        plt.xlabel("Mixing ratio")
        plt.ylabel("Pressure (bar)")
        plt.tight_layout()
        box = ax.get_position()
        ax.set_position(
            [
                box.x0,
                box.y0,
                box.width * 0.8,
                box.height,
            ]
        )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=1,
            prop={"size": 11},
            frameon=False,
        )
        if self.title:
            plt.title(self.title + " - Active", fontsize=14)
        plt.savefig(
            os.path.join(
                self.out_folder,
                "%s_fit_active_mixratio.pdf" % (self.prefix),
            )
        )
        plt.close()

        cols_mol = {}

        fig = plt.figure(figsize=(7, 7 / self.phi))
        ax = fig.add_subplot(111)
        num_moles = len(self.inactiveGases)

        for mol_idx, mol_name in enumerate(self.inactiveGases):
            cols_mol[mol_name] = self.cmap(mol_idx / num_moles)

            prof = inactive_profile[mol_idx]

            plt.plot(
                prof,
                pressure_profile,
                color=cols_mol[mol_name],
                label=mol_name,
            )

        plt.yscale("log")
        plt.gca().invert_yaxis()
        plt.xscale("log")
        plt.xlim(1e-12, 3)
        plt.xlabel("Mixing ratio")
        plt.ylabel("Pressure (bar)")
        plt.tight_layout()
        box = ax.get_position()
        ax.set_position(
            [
                box.x0,
                box.y0,
                box.width * 0.8,
                box.height,
            ]
        )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=1,
            prop={"size": 11},
            frameon=False,
        )
        if self.title:
            plt.title(self.title + " - Inactive", fontsize=14)
        plt.savefig(
            os.path.join(
                self.out_folder,
                "%s_fit_inactive_mixratio.pdf" % (self.prefix),
            )
        )
        plt.close()

    def plot_forward_cprofile(self):
        """Plot forward condensate profiles."""
        solution_val = self.forward_output()

        try:
            self.condensates
        except KeyError:
            print("No condensates in chemistry/file, ignoring plot")
            return

        profiles = solution_val["Profiles"]
        pressure_profile = profiles["pressure_profile"][:] / 1e5
        active_profile = profiles["condensate_profile"]

        cols_mol = {}

        fig = plt.figure(figsize=(7, 7 / self.phi))
        ax = fig.add_subplot(111)
        num_moles = len(self.condensates)

        for mol_idx, mol_name in enumerate(self.condensates):
            cols_mol[mol_name] = self.cmap(mol_idx / num_moles)

            prof = active_profile[mol_idx]

            plt.plot(
                prof,
                pressure_profile,
                color=cols_mol[mol_name],
                label=mol_name,
            )

        plt.yscale("log")
        plt.gca().invert_yaxis()
        plt.xscale("log")
        plt.xlim(1e-12, 3)
        plt.xlabel("Mixing ratio")
        plt.ylabel("Pressure (bar)")
        plt.tight_layout()
        box = ax.get_position()
        ax.set_position(
            [
                box.x0,
                box.y0,
                box.width * 0.8,
                box.height,
            ]
        )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=1,
            prop={"size": 11},
            frameon=False,
        )
        if self.title:
            plt.title(self.title, fontsize=14)
        plt.savefig(
            os.path.join(
                self.out_folder,
                "%s_fit_condensate_mixratio.pdf" % (self.prefix),
            )
        )
        plt.close()

    def plot_fitted_tp(self):
        """Plot fitted TP profile."""
        # fitted model
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(111)  # noqa: F841

        for solution_idx, solution_val in self.solution_iter():
            if self.num_solutions > 1:
                label = "Fitted profile (%i)" % (solution_idx)
            else:
                label = "Fitted profile"
            temp_prof = solution_val["Profiles"]["temp_profile"][:]
            temp_prof_std = solution_val["Profiles"]["temp_profile_std"][:]
            pres_prof = solution_val["Profiles"]["pressure_profile"][:] / 1e5
            plt.plot(
                temp_prof,
                pres_prof,
                color=self.cmap(float(solution_idx) / self.num_solutions),
                label=label,
            )
            plt.fill_betweenx(
                pres_prof,
                temp_prof - temp_prof_std,
                temp_prof + temp_prof_std,
                color=self.cmap(float(solution_idx) / self.num_solutions),
                alpha=0.5,
            )

        plt.yscale("log")
        plt.gca().invert_yaxis()
        plt.xlabel("Temperature (K)")
        plt.ylabel("Pressure (bar)")
        plt.tight_layout()
        legend = plt.legend(loc="upper left", ncol=1, prop={"size": 11})
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("white")

        legend.get_frame().set_alpha(0.8)
        if self.title:
            plt.title(self.title, fontsize=14)
        plt.savefig(
            os.path.join(
                self.out_folder,
                "%s_tp_profile.pdf" % (self.prefix),
            )
        )
        plt.close()

    def plot_forward_tp(self):
        """Plot forward TP profile."""
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(111)  # noqa: F841

        solution_val = self.forward_output()

        temp_prof = solution_val["Profiles"]["temp_profile"][:]
        pres_prof = solution_val["Profiles"]["pressure_profile"][:] / 1e5
        plt.plot(temp_prof, pres_prof)

        plt.yscale("log")
        plt.gca().invert_yaxis()
        plt.xlabel("Temperature (K)")
        plt.ylabel("Pressure (bar)")
        plt.tight_layout()
        legend = plt.legend(loc="upper left", ncol=1, prop={"size": 11})
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("white")

        legend.get_frame().set_alpha(0.8)
        if self.title:
            plt.title(self.title, fontsize=14)
        plt.savefig(
            os.path.join(
                self.out_folder,
                "%s_tp_profile.pdf" % (self.prefix),
            )
        )
        plt.close()

    def get_derived_parameters(self, solution):
        """Get derived parameters."""
        if "derived_params" in solution:
            return [c for _, c in solution["derived_params"].items()]
        else:
            return [solution["fit_params"]["mu_derived"]]

    def plot_posteriors(
        self,
        fig=None,
        save=True,
        ranges=None,
        plot_mu=True,
        color=None,
        truth=None,
        selected_fitparams=None,
    ):
        """Plot posteriors."""
        if not self.is_retrieval:
            raise Exception(
                "HDF5 was not generated from retrieval, " "no posteriors found"
            )
        if selected_fitparams is None:
            selected_fitparams = self.fittingNames
        if ranges is None:
            ranges = self.compute_ranges(
                mu=plot_mu,
                selected_fitparams=selected_fitparams,
            )

        for solution_idx, solution_val in self.solution_iter():

            mu_derived = self.get_derived_parameters(solution_val)

            tracedata = solution_val["tracedata"]
            weights = solution_val["weights"]

            figure_past = fig

            indices = np.array([self.fittingNames.index(x) for x in selected_fitparams])
            _tracedata = tracedata[..., indices]
            latex_names = [self.fittingLatex[idx] for idx in indices]

            if mu_derived is not None:
                for param in mu_derived:

                    index = self.derivedNames.index(param.name.split("/")[-1])
                    latex_names.append(self.derivedLatex[index])
                    _tracedata = np.column_stack((_tracedata, param["trace"]))

            if color is None:
                color_idx = float(solution_idx) / self.num_solutions
                color = self.cmap(float(color_idx))

            # https://matplotlib.org/users/customizing.html
            plt.rc("xtick", labelsize=10)  # size of individual labels
            plt.rc("ytick", labelsize=10)
            plt.rc("axes.formatter", limits=(-4, 5))  # scientific notation..

            fig = corner.corner(
                _tracedata,
                weights=weights,
                labels=latex_names,
                label_kwargs=dict(fontsize=20),
                smooth=1.5,
                scale_hist=True,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs=dict(fontsize=12),
                range=ranges,
                truths=truth,
                # quantiles=[0.16, 0.5],
                ret=True,
                fill_contours=True,
                color=color,
                top_ticks=False,
                bins=100,
                fig=figure_past,
            )
            if self.title:
                fig.gca().annotate(
                    self.title,
                    xy=(0.5, 1.0),
                    xycoords="figure fraction",
                    xytext=(0, -5),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=14,
                )
        if save:
            plt.savefig(
                os.path.join(
                    self.out_folder,
                    "%s_posteriors.pdf" % (self.prefix),
                )
            )
            plt.close()
        else:
            return fig

    @property
    def modelType(self):
        """Get model type."""
        return self.fd["ModelParameters"]["model_type"][()]

    def plot_fitted_spectrum(self, resolution=None):
        """Plot fitted spectrum."""
        # fitted model
        fig = plt.figure(figsize=(10.6, 7.0))
        # ax = fig.add_subplot(111)
        fig = fig  # noqa: F841

        obs_spectrum = self.fd["Observed"]["spectrum"][...]
        error = self.fd["Observed"]["errorbars"][...]
        wlgrid = self.fd["Observed"]["wlgrid"][...]

        plt.errorbar(
            wlgrid,
            obs_spectrum,
            error,
            lw=1,
            color="black",
            alpha=0.4,
            ls="none",
            zorder=0,
            label="Observed",
        )

        for solution_idx, solution_val in self.solution_iter():
            if self.num_solutions > 1:
                label = "Fitted model (%i)" % (solution_idx)
            else:
                label = "Fitted model"

            try:
                binned_grid = solution_val["Spectra"]["binned_wlgrid"][...]
            except KeyError:
                binned_grid = solution_val["Spectra"]["bin_wlgrid"][...]

            native_grid = solution_val["Spectra"]["native_wngrid"][...]

            plt.scatter(
                wlgrid,
                obs_spectrum,
                marker="d",
                zorder=1,
                **{
                    "s": 10,
                    "edgecolors": "grey",
                    "c": self.cmap(float(solution_idx) / self.num_solutions),
                },
            )

            self._generic_plot(
                binned_grid,
                native_grid,
                solution_val["Spectra"],
                resolution=resolution,
                color=self.cmap(float(solution_idx) / self.num_solutions),
                label=label,
            )

        plt.xlim(
            np.min(wlgrid) - 0.05 * np.min(wlgrid),
            np.max(wlgrid) + 0.05 * np.max(wlgrid),
        )
        # plt.ylim(0.0,0.006)
        plt.xlabel(r"Wavelength ($\mu$m)")
        try:
            plt.ylabel(self.model_axis[self.modelType])
        except KeyError:
            pass

        if np.max(wlgrid) - np.min(wlgrid) > 5:
            plt.xscale("log")
            plt.tick_params(axis="x", which="minor")
            # ax.xaxis.set_minor_formatter(
            #     mpl.ticker.FormatStrFormatter("%i")
            # )
            # ax.xaxis.set_major_formatter(
            #     mpl.ticker.FormatStrFormatter("%i")
            # )
        plt.legend(
            loc="best",
            ncol=2,
            frameon=False,
            prop={"size": 11},
        )
        if self.title:
            plt.title(self.title, fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.out_folder,
                "%s_spectrum.pdf" % (self.prefix),
            )
        )
        plt.close()

    def plot_forward_spectrum(self, resolution=None):
        """Plot forward spectrum."""
        fig = plt.figure(figsize=(5.3, 3.5))
        fig = fig  # noqa: F841

        spectra_out = self.forward_output()["Spectra"]

        native_grid = spectra_out["native_wngrid"][...]

        try:
            wlgrid = spectra_out["binned_wlgrid"][...]
        except KeyError:
            wlgrid = spectra_out["native_wlgrid"][...]

        self._generic_plot(
            wlgrid,
            native_grid,
            spectra_out,
            resolution=resolution,
            alpha=1,
        )
        plt.xlim(
            np.min(wlgrid) - 0.05 * np.min(wlgrid),
            np.max(wlgrid) + 0.05 * np.max(wlgrid),
        )
        # plt.ylim(0.0,0.006)
        plt.xlabel(r"Wavelength ($\mu$m)")
        try:
            plt.ylabel(self.model_axis[self.modelType])
        except KeyError:
            pass

        if np.max(wlgrid) - np.min(wlgrid) > 5:
            plt.xscale("log")
            plt.tick_params(axis="x", which="minor")
            # ax.xaxis.set_minor_formatter(
            #     mpl.ticker.FormatStrFormatter("%i")
            # )
            # ax.xaxis.set_major_formatter(
            #     mpl.ticker.FormatStrFormatter("%i")
            # )
        plt.legend(
            loc="best",
            ncol=2,
            frameon=False,
            prop={"size": 11},
        )
        if self.title:
            plt.title(self.title, fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.out_folder,
                "%s_forward_spectrum.pdf" % (self.prefix),
            )
        )
        plt.close()

    def plot_fitted_contrib(self, full=False, resolution=None):
        """Plot fitted contributions."""
        # fitted model

        for solution_idx, solution_val in self.solution_iter():

            fig = plt.figure(figsize=(5.3 * 2, 3.5 * 2))
            ax = fig.add_subplot(111)

            obs_spectrum = self.fd["Observed"]["spectrum"][:]
            error = self.fd["Observed"]["errorbars"][...]
            wlgrid = self.fd["Observed"]["wlgrid"][...]

            plt.errorbar(
                wlgrid,
                obs_spectrum,
                error,
                lw=1,
                color="black",
                alpha=0.4,
                ls="none",
                zorder=0,
                label="Observed",
            )
            self._plot_contrib(
                solution_val,
                wlgrid,
                ax,
                full=full,
                resolution=resolution,
            )

            # plt.tight_layout()
            plt.savefig(
                os.path.join(
                    self.out_folder,
                    "%s_spectrum_contrib_sol%i.pdf" % (self.prefix, solution_idx),
                )
            )
            plt.close()

        plt.close("all")

    def plot_forward_contrib(self, full=False, resolution=None):
        """Plot forward contributions."""
        fig = plt.figure(figsize=(5.3 * 2, 3.5 * 2))
        ax = fig.add_subplot(111)

        spectra_out = self.forward_output()["Spectra"]

        native_grid = spectra_out["native_wngrid"][...]

        try:
            wlgrid = spectra_out["binned_wlgrid"][...]
        except KeyError:
            wlgrid = spectra_out["native_wlgrid"][...]

        self._generic_plot(
            wlgrid,
            native_grid,
            spectra_out,
            resolution=resolution,
            alpha=0.5,
        )
        self._plot_contrib(
            self.forward_output(),
            wlgrid,
            ax,
            full=full,
            resolution=resolution,
        )

        # plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.out_folder,
                "%s_spectrum_contrib_forward.pdf" % (self.prefix),
            )
        )
        plt.close()

    def _plot_contrib(
        self,
        output,
        wlgrid,
        ax,
        full=False,
        resolution=None,
    ):
        """Plot contributions."""
        if full:
            wlgrid = self.full_contrib_plot(
                output["Spectra"],
                wlgrid,
                resolution=resolution,
            )
        else:
            wlgrid = self.simple_contrib_plot(
                output["Spectra"],
                wlgrid,
                resolution=resolution,
            )

        plt.xlim(
            np.min(wlgrid) - 0.05 * np.min(wlgrid),
            np.max(wlgrid) + 0.05 * np.max(wlgrid),
        )
        # plt.ylim(0.0,0.006)
        plt.xlabel(r"Wavelength ($\mu$m)")
        try:
            plt.ylabel(self.model_axis[self.modelType])
        except KeyError:
            pass

        if np.max(wlgrid) - np.min(wlgrid) > 5:
            plt.xscale("log")
            plt.tick_params(axis="x", which="minor")
            # ax.xaxis.set_minor_formatter(
            #     mpl.ticker.FormatStrFormatter("%i")
            # )
            # ax.xaxis.set_major_formatter(
            #     mpl.ticker.FormatStrFormatter("%i")
            # )
        # plt.legend(loc='best', ncol=2, frameon=False,
        #             prop={'size':11})
        box = ax.get_position()
        ax.set_position(
            [
                box.x0,
                box.y0 + box.height * 0.1,
                box.width,
                box.height * 0.9,
            ]
        )
        # Put a legend below current axis
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            fancybox=True,
            shadow=True,
            ncol=5,
        )
        if self.title:
            plt.title(self.title, fontsize=14)

    def full_contrib_plot(self, spectra, wlgrid, resolution=None):
        """Plot full contributions."""
        native_grid = spectra["native_wngrid"][...]
        for (
            contrib_name,
            contrib_dict,
        ) in spectra["Contributions"].items():

            for (
                component_name,
                component_value,
            ) in contrib_dict.items():
                if isinstance(component_value, h5py.Dataset):
                    continue
                total_label = f"{contrib_name}-{component_name}"
                self._generic_plot(
                    wlgrid,
                    native_grid,
                    component_value,
                    resolution,
                    label=total_label,
                )
        return wlgrid

    def simple_contrib_plot(self, spectra, wlgrid, resolution=None):
        """Plot simple contributions."""
        native_grid = spectra["native_wngrid"][...]

        for (
            contrib_name,
            contrib_dict,
        ) in spectra["Contributions"].items():
            if contrib_name == "Absorption":
                for (
                    component_name,
                    component_value,
                ) in contrib_dict.items():
                    if isinstance(component_value, h5py.Dataset):
                        continue
                    total_label = f"{contrib_name}-{component_name}"
                    self._generic_plot(
                        wlgrid,
                        native_grid,
                        component_value,
                        resolution,
                        label=total_label,
                    )
            else:
                self._generic_plot(
                    wlgrid,
                    native_grid,
                    contrib_dict,
                    resolution,
                )

        return wlgrid

    def _generic_plot(  # noqa: C901
        self,
        wlgrid,
        native_grid,
        spectra,
        resolution,
        color=None,
        error=False,
        alpha=1.0,
        label=None,
    ):
        """Generic plotting function."""
        binned_error = None
        if resolution is not None:
            from taurex.binning import FluxBinner
            from taurex.util.util import create_grid_res
            from taurex.util.util import wnwidth_to_wlwidth

            _grid = create_grid_res(
                resolution,
                wlgrid.min() * 0.9,
                wlgrid.max() * 1.1,
            )
            bin_wlgrid = _grid[:, 0]

            bin_wngrid = 10000 / _grid[:, 0]

            bin_sort = bin_wngrid.argsort()

            bin_wlgrid = bin_wlgrid[bin_sort]
            bin_wngrid = bin_wngrid[bin_sort]

            bin_wnwidth = wnwidth_to_wlwidth(bin_wlgrid, _grid[bin_sort, 1])
            wlgrid = _grid[bin_sort, 0]
            binner = FluxBinner(bin_wngrid, bin_wnwidth)
            native_spectra = spectra["native_spectrum"][...]
            binned_spectrum = binner.bindown(native_grid, native_spectra)[1]
            try:
                native_error = spectra["native_std"]
            except KeyError:
                native_error = None
            if native_error is not None:
                binned_error = binner.bindown(native_grid, native_error)[1]

        else:
            try:
                binned_spectrum = spectra["binned_spectrum"][...]
            except KeyError:
                try:
                    binned_spectrum = spectra["bin_spectrum"][...]
                except KeyError:
                    binned_spectrum = spectra["native_spectrum"][...]
            try:
                binned_error = spectra["binned_std"][...]
            except KeyError:
                binned_error = None

        good = binned_spectrum > 0
        plt.plot(
            wlgrid[good],
            binned_spectrum[good],
            label=label,
            alpha=alpha,
        )
        if binned_error is not None:
            plt.fill_between(
                wlgrid[good],
                binned_spectrum[good] - binned_error[good],
                binned_spectrum[good] + binned_error[good],
                alpha=0.5,
                zorder=-2,
                color=color,
                edgecolor="none",
            )

            # 2 sigma
            plt.fill_between(
                wlgrid[good],
                binned_spectrum[good] - 2 * binned_error[good],
                binned_spectrum[good] + 2 * binned_error[good],
                alpha=0.2,
                zorder=-3,
                color=color,
                edgecolor="none",
            )

    def close(self):
        """Close HDF5 file."""
        self.fd.close()

    def plot_forward_tau(self):
        """Plot forward tau."""
        forward_output = self.forward_output()

        contribution = forward_output["Spectra"]["native_tau"][...]
        # contribution = self.pickle_file['solutions'][
        #     solution_idx
        # ]['contrib_func']

        pressure = forward_output["Profiles"]["pressure_profile"][:]
        wavelength = forward_output["Spectra"]["native_wlgrid"][:]

        self._plot_tau(contribution, pressure, wavelength)

        plt.savefig(
            os.path.join(
                self.out_folder,
                "%s_tau_forward.pdf" % (self.prefix),
            )
        )

        plt.close()

    def plot_fitted_tau(self, wl_min: float = 0.5, wl_max: float = 12.0):
        """Plot fitted tau."""
        for solution_idx, solution_val in self.solution_iter():

            contribution = solution_val["Spectra"]["native_tau"][...]
            # contribution = self.pickle_file['solutions'][
            #     solution_idx
            # ]['contrib_func']

            pressure = solution_val["Profiles"]["pressure_profile"][:]
            wavelength = solution_val["Spectra"]["native_wlgrid"][:]

            wavelength_right = self.find_nearest(wavelength, wl_min)[0]
            wavelength_left = self.find_nearest(wavelength, wl_max)[0]

            wavelength = wavelength[wavelength_left : wavelength_right + 1]
            contribution = contribution[
                ...,
                wavelength_left : wavelength_right + 1,
            ]

            self._plot_tau(contribution, pressure, wavelength)

            plt.savefig(
                os.path.join(
                    self.out_folder,
                    "%s_tau_sol%i.pdf" % (self.prefix, solution_idx),
                )
            )

            plt.close()

    def _plot_tau(self, contribution, pressure, wavelength):
        """Plot tau."""
        grid = plt.GridSpec(1, 4, wspace=0.4, hspace=0.3)
        fig = plt.figure("Contribution function")
        fig = fig  # noqa: F841
        ax1 = plt.subplot(grid[0, :3])
        pos = plt.imshow(contribution, aspect="auto")
        plt.colorbar(pos, ax=ax1)

        # mapping of the pressure array onto the ticks:
        y_labels = np.array(
            [
                pow(10.0, p)
                for p in np.arange(
                    np.ceil(np.log10(np.max(pressure))),
                    np.floor(np.log10(np.min(pressure))) - 1.0,
                    step=-1,
                )
            ]
        )
        y_ticks = np.zeros(len(y_labels))
        for i in range(len(y_ticks)):
            y_ticks[i] = (
                np.abs(pressure - y_labels[i])
            ).argmin()  # To find the corresponding index
        plt.yticks(
            y_ticks,
            ["$10^{%.f}$" % y for y in np.log10(y_labels) - 5],
        )

        # mapping of the wavelength array onto the ticks:
        x_label0 = np.ceil(np.min(wavelength) * 10) / 10.0
        x_label5 = np.round(np.max(wavelength) * 10) / 10.0
        x_label1 = (
            np.round(
                pow(
                    10,
                    (np.log10(x_label5) - np.log10(x_label0)) * 1 / 5.0
                    + np.log10(x_label0),
                )
                * 10
            )
            / 10.0
        )
        x_label2 = (
            np.round(
                pow(
                    10,
                    (np.log10(x_label5) - np.log10(x_label0)) * 2 / 5.0
                    + np.log10(x_label0),
                )
                * 10
            )
            / 10.0
        )
        x_label3 = (
            np.round(
                pow(
                    10,
                    (np.log10(x_label5) - np.log10(x_label0)) * 3 / 5.0
                    + np.log10(x_label0),
                )
                * 10
            )
            / 10.0
        )
        x_label4 = (
            np.round(
                pow(
                    10,
                    (np.log10(x_label5) - np.log10(x_label0)) * 4 / 5.0
                    + np.log10(x_label0),
                )
                * 10
            )
            / 10.0
        )

        x_labels = np.array(
            [
                x_label0,
                x_label1,
                x_label2,
                x_label3,
                x_label4,
                x_label5,
            ]
        )
        x_ticks = np.zeros(len(x_labels))
        for i in range(len(x_ticks)):
            x_ticks[i] = (
                np.abs(wavelength - x_labels[i])
            ).argmin()  # To find the corresponding index
        plt.xticks(x_ticks, x_labels)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.xlabel(r"Wavelength [$\mu$m]")
        plt.ylabel("Pressure [bar]")

        ax2 = plt.subplot(grid[0, 3])
        ax2 = ax2  # noqa: F841

        contribution_collapsed = np.average(contribution, axis=1)
        # contribution_collapsed = np.amax(
        #     contribution_hr, axis=1
        # )
        # good for emission
        contribution_sum = np.zeros(len(contribution_collapsed))
        for i in range(len(contribution_collapsed) - 1):
            contribution_sum[i + 1] = (
                contribution_sum[i] + contribution_collapsed[i + 1]
            )
        plt.plot(
            contribution_collapsed,
            pressure * pow(10, -5),
        )
        plt.ylim(y_labels[0] / 1.0e5, y_labels[-1] / 1.0e5)
        plt.yscale("log")
        plt.gca().yaxis.tick_right()
        plt.xlabel("Contribution")

    @property
    def fittingNames(self):
        """Get fitting names."""
        from taurex.util.util import decode_string_array

        if not self.is_retrieval:
            raise Exception(
                "HDF5 was not generated from retrieval, " "no fitting names found"
            )
        return decode_string_array(self.fd["Optimizer"]["fit_parameter_names"])

    @property
    def fittingLatex(self):
        """Get fitting latex."""
        from taurex.util.util import decode_string_array

        if not self.is_retrieval:
            raise Exception(
                "HDF5 was not generated from retrieval, " "no fitting latex found"
            )
        return decode_string_array(self.fd["Optimizer"]["fit_parameter_latex"])

    @property
    def derivedNames(self):
        """Get derived names."""
        from taurex.util.util import decode_string_array

        if not self.is_retrieval:
            raise Exception(
                "HDF5 was not generated from retrieval, " "no fitting latex found"
            )
        try:
            array = decode_string_array(self.fd["Optimizer"]["derived_parameter_names"])
            return [f"{c}_derived" for c in array]
        except KeyError:
            return ["mu_derived"]

    @property
    def derivedLatex(self):
        """Get derived latex."""
        from taurex.util.util import decode_string_array

        if not self.is_retrieval:
            raise Exception(
                "HDF5 was not generated from retrieval, " "no fitting latex found"
            )
        try:
            array = decode_string_array(self.fd["Optimizer"]["derived_parameter_latex"])
            return [f"{c} (derived)" for c in array]
        except KeyError:
            return [r"$\mu$ (derived)"]

    @property
    def fittingBoundaryLow(self):
        """Get fitting boundary low."""
        if not self.is_retrieval:
            raise Exception(
                "HDF5 was not generated from retrieval, " "no fitting boundary found"
            )
        return self.fd["Optimizer"]["fit_boundary_low"][:]

    @property
    def fittingBoundaryHigh(self):
        """Get fitting boundary high."""
        if not self.is_retrieval:
            raise Exception(
                "HDF5 was not generated from retrieval, " "no fitting boundary found"
            )
        return self.fd["Optimizer"]["fit_boundary_high"][:]

    @property
    def is_retrieval(self):
        """Check if retrieval."""
        try:
            self.fd["Output"]
            self.fd["Optimizer"]
            self.fd["Output"]["Solutions"]
            return True
        except KeyError:
            return False

    @property
    def is_lightcurve(self):
        """Check if lightcurve."""
        try:
            self.fd["Lightcurve"]
            return True
        except KeyError:
            return False


def main():  # noqa: C901
    """Plotter main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Taurex-Plotter")
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input hdf5 file from taurex",
    )
    parser.add_argument(
        "-P",
        "--plot-posteriors",
        dest="posterior",
        default=False,
        help="Plot fitting posteriors",
        action="store_true",
    )
    parser.add_argument(
        "-x",
        "--plot-xprofile",
        dest="xprofile",
        default=False,
        help="Plot molecular profiles",
        action="store_true",
    )
    parser.add_argument(
        "-D",
        "--plot-cprofile",
        dest="cprofile",
        default=False,
        help="Plot condensate profiles",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--plot-tpprofile",
        dest="tpprofile",
        default=False,
        help="Plot Temperature profiles",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--plot-tau",
        dest="tau",
        default=False,
        help="Plot optical depth contribution",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--plot-spectrum",
        dest="spectrum",
        default=False,
        help="Plot spectrum",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--plot-contrib",
        dest="contrib",
        default=False,
        help="Plot contrib",
        action="store_true",
    )
    parser.add_argument(
        "-C",
        "--full-contrib",
        dest="full_contrib",
        default=False,
        help="Plot detailed contribs",
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--all",
        dest="all",
        default=False,
        help="Plot everythiong",
        action="store_true",
    )
    parser.add_argument(
        "-T",
        "--title",
        dest="title",
        type=str,
        help="Title of plots",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        required=True,
        help="output directory to store plots",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        dest="prefix",
        type=str,
        help="File prefix for outputs",
    )
    parser.add_argument(
        "-m",
        "--color-map",
        dest="cmap",
        type=str,
        default="Paired",
        help="Matplotlib colormap to use",
    )
    parser.add_argument(
        "-R",
        "--resolution",
        dest="resolution",
        type=float,
        default=None,
        help="Resolution to bin spectra to",
    )
    args = parser.parse_args()

    plot_xprofile = args.xprofile or args.all
    plot_tp_profile = args.tpprofile or args.all
    plot_spectrum = args.spectrum or args.all
    plot_contrib = args.contrib or args.all
    plot_fullcontrib = args.full_contrib or args.all
    plot_posteriors = args.posterior or args.all
    plot_tau = args.tau or args.all
    plot_cond = args.cprofile or args.all

    plot = Plotter(
        args.input_file,
        cmap=args.cmap,
        title=args.title,
        prefix=args.prefix,
        out_folder=args.output_dir,
    )

    if plot_posteriors:
        if plot.is_retrieval:
            plot.plot_posteriors()

    if plot_xprofile:
        if plot.is_retrieval:
            plot.plot_fit_xprofile()
        else:
            plot.plot_forward_xprofile()
    if plot_spectrum:
        if plot.is_retrieval:
            plot.plot_fitted_spectrum(resolution=args.resolution)
        else:
            plot.plot_forward_spectrum(resolution=args.resolution)
    if plot_tp_profile:
        if plot.is_retrieval:
            plot.plot_fitted_tp()
        else:
            plot.plot_forward_tp()

    if plot_cond:
        if plot.is_retrieval:
            pass
        else:
            plot.plot_forward_cprofile()

    if plot_contrib:
        if plot.is_retrieval:
            plot.plot_fitted_contrib(
                full=plot_fullcontrib,
                resolution=args.resolution,
            )
        else:
            plot.plot_forward_contrib(
                full=plot_fullcontrib,
                resolution=args.resolution,
            )

    if plot_tau:
        if plot.is_retrieval:
            plot.plot_fitted_tau()
        else:
            plot.plot_forward_tau()


if __name__ == "__main__":
    main()
