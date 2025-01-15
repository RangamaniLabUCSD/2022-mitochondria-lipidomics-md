# %%

# %%
import pickle
import numpy as np
from functools import partial
import MDAnalysis

from pathlib import Path

import matplotlib.pyplot as plt
import numpy.typing as npt

import pandas as pd

from scipy import integrate, interpolate, stats, signal
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

import util
from plot_helper import *


# %%
plot_styles = [
    ("./white_background_ctl.mplstyle", ""),
    ("./dark_background_ctl.mplstyle", "_dark"),
]


# %%
from MDAnalysis.analysis.leaflet import LeafletFinder


def largest_groups(atoms):
    """
    From a list of sizes, find out the indices of the two largest groups. These should correspond to the two leaflets of the bilayer.

    Keyword arguments:
    atoms -- list of sizes of clusters identified by LeafletFinder
    """
    largest = 0
    second_largest = 0

    for i in atoms:
        if atoms[i] > largest:
            largest_index = i
            largest = atoms[i]

    for i in atoms:
        if atoms[i] > second_largest and i != largest_index:
            second_largest_index = i
            second_largest = atoms[i]

    return (largest_index, second_largest_index)


def determine_leaflets(universe, selection="all"):
    """
    From a selection of phosphates, determine which belong to the upper and lower leaflets.

    Keyword arguments:
    universe -- an MDAnalysis Universe object
    phosphateSelection -- a string specifying how to select the phosphate atoms (e.g. "name P1")
    """
    leaflets = {}

    # calculate the z value of the phosphates defining the bilayer (assumes bilayer is in x and y..)
    ag = universe.atoms.select_atoms(selection)
    bilayerCentre = ag.center_of_geometry()[2]

    # apply the MDAnalysis LeafletFinder graph-based method to determine the two largest groups which
    #  should correspond to the upper and lower leaflets
    phosphates = LeafletFinder(universe, selection)

    # find the two largest groups - required as the first two returned by LeafletFinder, whilst usually are the largest, this is not always so
    (a, b) = largest_groups(phosphates.sizes())

    # check to see where the first leaflet lies
    if phosphates.group(a).centroid()[2] > bilayerCentre:
        leaflets["upper"] = phosphates.group(a)
        leaflets["lower"] = phosphates.group(b)
    else:
        leaflets["lower"] = phosphates.group(a)
        leaflets["upper"] = phosphates.group(b)

    return leaflets


# %%
lipids = ["POPC", "DOPC", "POPE", "DOPE", "CDL1", "CDL2", "POPG", "DOPG"]
leaflets = ["upper", "lower"]

queries = {
    "PC": "resname POPC or resname DOPC",
    "PE": "resname POPE or resname DOPE",
    "CDL": "resname CDL1 or resname CDL2",
    "PG": "resname POPG or resname DOPG",
}

po4_neighbor_sel = "name PO4 or name GL0"


def get_midpoints(x):
    return ((x + np.roll(x, -1)) / 2)[:-1]


# %%
def fftAutocovariance(signal):
    """
    FFT based calculation of the autocovariance function <df(0)*df(t)> without wrapping
    """
    centered_signal = signal - np.mean(signal, axis=0)
    zero_padding = np.zeros_like(centered_signal)
    padded_signal = np.concatenate((centered_signal, zero_padding))
    ft_signal = np.fft.fft(padded_signal, axis=0)
    pseudo_autocovariance = np.fft.ifft(np.abs(ft_signal) ** 2, axis=0)
    input_domain = np.ones_like(centered_signal)
    mask = np.concatenate((input_domain, zero_padding))
    ft_mask = np.fft.fft(mask, axis=0)
    mask_correction_factors = np.fft.ifft(np.abs(ft_mask) ** 2, axis=0)
    autocovariance = pseudo_autocovariance / mask_correction_factors
    return np.real(autocovariance[0 : len(signal)])


def fftAutocorrelation(signal):
    """
    FFT calculation of the normalized autocorrelation <df(0)*df(t)>/var(f) without wrapping
    """
    autocovariance = fftAutocovariance(signal)
    variance = autocovariance[0]
    # if variance == 0.:
    #     return np.zeros(autocovariance.shape)
    # else:
    return autocovariance / variance

def wrap_and_sanitize(pxy, ts, mc):
    """Wrap coordinates and remove values too far from closest known point

    Args:
        pxy (np.ndarray): XY coordinates
        ts: MDAnalysis timestep object
        mc: curvature object

    Returns:
        _type_: _description_
    """
    gx = mc.x_range[1] - mc.x_step
    
    # wrap PBC if point is more than half step greater than the closest data value
    if ts.dimensions[0] > mc.x_range[1] - mc.x_step / 2:
        pxy = np.where(pxy > gx + mc.x_step / 2, pxy - ts.dimensions[0], pxy)
    # Remove values which are too far from a known data point
    pxy = pxy[(pxy >= -mc.x_step / 2).all(axis=1), :]
    return pxy


# %%
show_figs = False
curr_fig_path = Path("Figures/curvature_spatial_autocorrelation")
curr_fig_path.mkdir(parents=True, exist_ok=True)

time_autocorrelations = {}

for sim in util.simulations:
    # for sim in ["1"]:
    print(sim)
    with open(
        util.analysis_path / f"{sim}/membrane_curvature_2nm.pickle", "rb"
    ) as handle:
        mc = pickle.load(handle)

    h = mc.results["height"][1:]
    mean_curvatures = np.zeros_like(h)
    for i in range(h.shape[0]):
        mean_curvatures[i] = util.mean_curvature(h[i], mc.x_step)

    mean_curvatures -= np.mean(mean_curvatures)

    acr = fftAutocorrelation(mean_curvatures)
    height_acr = fftAutocorrelation(h)
    # print(acr.shape)
    acr = np.mean(acr, axis=(1, 2))  # get mean over axis 1 and 2
    height_acr = np.mean(height_acr, axis=(1, 2))

    if sim != "1_vbt":
        time_autocorrelations[util.sim_to_final_index[int(sim)]] = acr

    for style, style_ext in plot_styles:
        with plt.style.context(style):
            if style_ext:
                ecolor = "white"
            else:
                ecolor = "black"

            fig, ax = plt.subplots(1, 1, figsize=(3, 3))  # sharex=True,

            ax.plot(np.arange(0, 100), acr[0:100])

            ax.set_ylabel("Time Autocorrelation")
            ax.set_xlabel("Frame Lag")

            fig.tight_layout()

            if sim == "1_vbt":
                save_fig(
                    fig, curr_fig_path / f"1_vbt_curvature_localization_time{style_ext}"
                )
            else:
                save_fig(
                    fig,
                    curr_fig_path
                    / f"{util.sim_to_final_index[int(sim)]}_curvature_localization_time{style_ext}",
                )

            if show_figs:
                plt.show()

            fig.clear()
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(3, 3))  # sharex=True,

            ax.plot(np.arange(0, 100), height_acr[0:100])

            ax.set_ylabel("Time Autocorrelation")
            ax.set_xlabel("Frame Lag")

            fig.tight_layout()

            if sim == "1_vbt":
                save_fig(
                    fig, curr_fig_path / f"1_vbt_height_localization_time{style_ext}"
                )
            else:
                save_fig(
                    fig,
                    curr_fig_path
                    / f"{util.sim_to_final_index[int(sim)]}_height_localization_time{style_ext}",
                )

            if show_figs:
                plt.show()

            fig.clear()
            plt.close(fig)

    ## COMPUTE 2D SPATIAL CORRELATION
    acr = []
    for i in tqdm(range(1, 100)):
        acr.append(
            signal.correlate2d(
                mean_curvatures[-i],
                mean_curvatures[-i],
                boundary="wrap",
            )
            / mean_curvatures.shape[1] ** 2
            / np.std(mean_curvatures[-i]) ** 2
        )

    acr = np.mean(np.array(acr), axis=0)

    for style, style_ext in plot_styles:
        with plt.style.context(style):
            if style_ext:
                ecolor = "white"
            else:
                ecolor = "black"

            fig, ax = plt.subplots(1, 1, figsize=(3, 3))  # sharex=True,

            shape_size = mc.x_range[1]/10
            im = ax.imshow(
                acr,
                vmin=-1,
                vmax=1,
                extent=[-shape_size, shape_size, -shape_size, shape_size],
            )

            fig.colorbar(im, ax=ax)
            ax.set_ylabel("Y (nm)")
            ax.set_xlabel("X (nm)")

            # limits = (-10, 10)
            # ax.set_xlim(*limits)
            # ax.set_ylim(*limits)

            fig.tight_layout()

            if sim == "1_vbt":
                save_fig(
                    fig,
                    curr_fig_path / f"1_vbt_curvature_localization_space{style_ext}",
                )
            else:
                save_fig(
                    fig,
                    curr_fig_path
                    / f"{util.sim_to_final_index[int(sim)]}_curvature_localization_space{style_ext}",
                )

            if show_figs:
                plt.show()

            fig.clear()
            plt.close(fig)


# %%
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


light_factor = 0.8

p = sns.color_palette("colorblind")

palette = [
    lighten_color(p[i], j)
    for i, j in [
        (7, 1),
        (7, light_factor),
        (8, 1),
        (8, light_factor),
        (0, 1),
        (0, light_factor),
        (2, 1),
        (2, light_factor),
    ]
]

sns.palplot(palette)
sns.palplot(p)


# %%
show_figs = False
curr_fig_path = Path("Figures/curvature_spatial_autocorrelation")
curr_fig_path.mkdir(parents=True, exist_ok=True)

color_index = [
    0,
    1,
    1,
    1,
    3,
    3,
    3,
    2,
    2,
    2,
    1,
    1,
    3,
    3,
    2,
    2,
    1,
    1,
    3,
    3,
    1,
    3,
]

# cmap = mpl.cm.get_cmap("viridis")
# c = cmap(np.linspace(0, 1, 22))

for style, style_ext in plot_styles:
    with plt.style.context(style):
        if style_ext:
            ecolor = "white"
        else:
            ecolor = "black"

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))  # sharex=True,
        for i in range(1, 22):
            ax.plot(
                np.arange(0, 100) * 500 / 1000,
                time_autocorrelations[i][0:100],
                color=p[color_index[i]],
                label=color_index[i],
                alpha=0.5,
            )

        ax.set_ylabel("Time Autocorrelation")
        ax.set_xlabel("Frame Lag (ns)")

        ax.axvline(5*500/1000, color=ecolor, linestyle="--", linewidth=0.5)

        # Creating legend with color box
        patches = [
            mpatches.Patch(color=p[1], label="CDL1"),
            mpatches.Patch(color=p[3], label="CDL2"),
            mpatches.Patch(color=p[2], label="PG"),
        ]
        ax.legend(handles=patches, loc="upper right")

        fig.tight_layout()

        save_fig(fig, curr_fig_path / f"all_time{style_ext}")

        if show_figs:
            plt.show()

        fig.clear()
        plt.close(fig)


# %%
show_figs = False
curr_fig_path = Path("Figures/geometric_localization")
curr_fig_path.mkdir(parents=True, exist_ok=True)

frames_to_average = 5

for sim in np.concatenate((util.simulations, ["1_vbt"])):
    print(f"sim {sim}")
    with open(
        util.analysis_path / f"{sim}/membrane_curvature_2nm.pickle", "rb"
    ) as handle:
        mc = pickle.load(handle)

    h = mc.results["height"][1:]
    mean_curvatures = np.zeros_like(h)
    for i in range(h.shape[0]):
        mean_curvatures[i] = util.mean_curvature(h[i], mc.x_step)

    mean_curvature_averaged = np.mean(
        mean_curvatures[-frames_to_average * 3 : -frames_to_average * 2], axis=0
    )

    gro = util.analysis_path / f"{sim}/po4_only.gro"
    traj = util.analysis_path / f"{sim}/po4_all.xtc"

    u = MDAnalysis.Universe(gro, str(traj), refresh_offsets=True)
    ag = determine_leaflets(u, po4_neighbor_sel)

    histograms = {}

    for k, query in queries.items():
        print(f"analyzing lipid {k}")

        if len(u.select_atoms(query)) == 0:
            continue

        upper_positions = []
        lower_positions = []
        for ts in u.trajectory[-frames_to_average:]:
            xy = wrap_and_sanitize(
                ag["upper"].select_atoms(query).positions[:, 0:2], ts, mc
            )
            upper_positions.append(xy)
            xy = wrap_and_sanitize(
                ag["lower"].select_atoms(query).positions[:, 0:2], ts, mc
            )
            lower_positions.append(xy)

        # Convert to numpy array
        upper_positions = np.vstack(upper_positions)
        lower_positions = np.vstack(lower_positions)

        H_upper, xe, ye = np.histogram2d(
            upper_positions[:, 0],
            upper_positions[:, 1],
            bins=mc.n_x_bins,
            range=[
                [mc.x_range[0] - mc.x_step / 2, mc.x_range[1] - mc.x_step / 2],
                [mc.y_range[0] - mc.x_step / 2, mc.y_range[1] - mc.x_step / 2],
            ],
            # density=True,
        )
        H_upper /= frames_to_average

        histograms[k] = (H_upper, xe, ye)

    for style, style_ext in plot_styles:
        with plt.style.context(style):
            if style_ext:
                ecolor = "white"
            else:
                ecolor = "black"

            # fig, ax = plt.subplots(1, 3, figsize=(9, 3))  # sharex=True,
            fig, ax = plt.subplots(
                1,
                len(histograms.keys()) + 1,
                figsize=(3 * (len(histograms.keys()) + 1), 3),
            )  # sharex=True,

            # NOTE reading XE etc from weird place...
            im0 = ax[0].imshow(
                mean_curvature_averaged * 10,
                extent=[xe[0] / 10, xe[-1] / 10, ye[0] / 10, ye[-1] / 10],
                interpolation="gaussian",
                origin="lower",
                cmap="PRGn",
            )
            ax[0].set_ylabel("Y (nm)")
            ax[0].set_xlabel("X (nm)")

            plt.colorbar(im0, ax=ax[0])

            ax[0].set_title(r"Mean Curvature (nm$^{-1}$)")

            vmax = 0
            for lipid, (H_upper, xe, ye) in histograms.items():
                if np.max(H_upper) > vmax:
                    vmax = np.max(H_upper)

            ax_index = 1
            for lipid, (H_upper, xe, ye) in histograms.items():
                im1 = ax[ax_index].imshow(
                    H_upper,
                    extent=[xe[0] / 10, xe[-1] / 10, ye[0] / 10, ye[-1] / 10],
                    interpolation="gaussian",
                    cmap="Purples",
                    vmin=0,
                    vmax=vmax,
                )
                plt.colorbar(im1, ax=ax[ax_index])
                ax[ax_index].set_title(f"{lipid} Localization")
                ax[ax_index].set_ylabel("Y (nm)")
                ax[ax_index].set_xlabel("X (nm)")
                # ax[2].scatter(upper_positions[:,0], upper_positions[:,1], color='k', s=0.1)
                # ax[2].set_aspect("equal")
                ax_index += 1

            if sim == "1_vbt":
                fig.suptitle(f"1_vbt")
            else:
                fig.suptitle(f"sim {util.sim_to_final_index[int(sim)]}")

            fig.tight_layout()

            if sim == "1_vbt":
                save_fig(
                    fig, curr_fig_path / f"1_vbt_curvature_localization{style_ext}"
                )
            else:
                save_fig(
                    fig,
                    curr_fig_path
                    / f"{util.sim_to_final_index[int(sim)]}_curvature_localization{style_ext}",
                )

            if show_figs:
                plt.show()

            fig.clear()
            plt.close(fig)


# %%
# show_figs = False
# curr_fig_path = Path("Figures/geometric_localization")
# curr_fig_path.mkdir(parents=True, exist_ok=True)

# frames_to_average = 5

# for sim in np.concatenate((util.simulations, ["1_vbt"])):
#     print(f"sim {sim}")
#     with open(
#         util.analysis_path / f"{sim}/membrane_curvature_2nm.pickle", "rb"
#     ) as handle:
#         mc = pickle.load(handle)

#     h = mc.results["height"][1:]
#     mean_curvatures = np.zeros_like(h)
#     for i in range(h.shape[0]):
#         mean_curvatures[i] = util.mean_curvature(h[i], mc.x_step)

#     mean_curvature_averaged = np.mean(
#         mean_curvatures[-frames_to_average * 3 : -frames_to_average * 2], axis=0
#     )

#     gro = util.analysis_path / f"{sim}/po4_only.gro"
#     traj = util.analysis_path / f"{sim}/po4_all.xtc"

#     u = MDAnalysis.Universe(gro, str(traj), refresh_offsets=True)
#     ag = determine_leaflets(u, po4_neighbor_sel)

#     histograms = {}

#     for k, query in queries.items():
#         print(f"analyzing lipid {k}")

#         if len(u.select_atoms(query)) == 0:
#             continue

#         upper_positions = []
#         lower_positions = []
#         for ts in u.trajectory[-frames_to_average:]:
#             xy = wrap_and_sanitize(
#                 ag["upper"].select_atoms(query).positions[:, 0:2], ts, mc
#             )
#             upper_positions.append(xy)
#             xy = wrap_and_sanitize(
#                 ag["lower"].select_atoms(query).positions[:, 0:2], ts, mc
#             )
#             lower_positions.append(xy)

#         # Convert to numpy array
#         upper_positions = np.vstack(upper_positions)
#         lower_positions = np.vstack(lower_positions)

#         H_upper, xe, ye = np.histogram2d(
#             upper_positions[:, 0],
#             upper_positions[:, 1],
#             bins=mc.n_x_bins,
#             range=[
#                 [mc.x_range[0] - mc.x_step / 2, mc.x_range[1] - mc.x_step / 2],
#                 [mc.y_range[0] - mc.x_step / 2, mc.y_range[1] - mc.x_step / 2],
#             ],
#             # density=True,
#         )
#         H_upper /= frames_to_average

#         histograms[k] = (H_upper, xe, ye)

#     for style, style_ext in plot_styles:
#         with plt.style.context(style):
#             if style_ext:
#                 ecolor = "white"
#             else:
#                 ecolor = "black"

#             # fig, ax = plt.subplots(1, 3, figsize=(9, 3))  # sharex=True,
#             fig, ax = plt.subplots(
#                 1,
#                 len(histograms.keys()) + 1,
#                 figsize=(3 * (len(histograms.keys()) + 1), 3),
#             )  # sharex=True,

#             # NOTE reading XE etc from weird place...
#             im0 = ax[0].imshow(
#                 mean_curvature_averaged * 10,
#                 extent=[xe[0] / 10, xe[-1] / 10, ye[0] / 10, ye[-1] / 10],
#                 interpolation="gaussian",
#                 origin="lower",
#                 cmap="PRGn",
#             )
#             ax[0].set_ylabel("Y (nm)")
#             ax[0].set_xlabel("X (nm)")

#             plt.colorbar(im0, ax=ax[0])

#             ax[0].set_title(r"Mean Curvature (nm$^{-1}$)")

#             vmax = 0
#             for lipid, (H_upper, xe, ye) in histograms.items():
#                 if np.max(H_upper) > vmax:
#                     vmax = np.max(H_upper)

#             ax_index = 1
#             for lipid, (H_upper, xe, ye) in histograms.items():
#                 im1 = ax[ax_index].imshow(
#                     H_upper,
#                     extent=[xe[0] / 10, xe[-1] / 10, ye[0] / 10, ye[-1] / 10],
#                     interpolation="gaussian",
#                     cmap="Purples",
#                     vmin=0,
#                     vmax=vmax,
#                 )
#                 plt.colorbar(im1, ax=ax[ax_index])
#                 ax[ax_index].set_title(f"{lipid} Localization")
#                 ax[ax_index].set_ylabel("Y (nm)")
#                 ax[ax_index].set_xlabel("X (nm)")
#                 # ax[2].scatter(upper_positions[:,0], upper_positions[:,1], color='k', s=0.1)
#                 # ax[2].set_aspect("equal")
#                 ax_index += 1

#             if sim == "1_vbt":
#                 fig.suptitle(f"1_vbt")
#             else:
#                 fig.suptitle(f"sim {util.sim_to_final_index[int(sim)]}")

#             fig.tight_layout()

#             if sim == "1_vbt":
#                 save_fig(
#                     fig, curr_fig_path / f"1_vbt_curvature_localization{style_ext}"
#                 )
#             else:
#                 save_fig(
#                     fig,
#                     curr_fig_path
#                     / f"{util.sim_to_final_index[int(sim)]}_curvature_localization{style_ext}",
#                 )

#             if show_figs:
#                 plt.show()

#             fig.clear()
#             plt.close(fig)


# %%
show_figs = True
curr_fig_path = Path("Figures/lipid_pearsons")
curr_fig_path.mkdir(parents=True, exist_ok=True)


frames_to_average = 5
sets_to_consider = 200


for sim in np.concatenate((util.simulations, ["1_vbt"])):
    print(f"sim {sim}")
    with open(
        util.analysis_path / f"{sim}/membrane_curvature_2nm.pickle", "rb"
    ) as handle:
        mc = pickle.load(handle)

    h = mc.results["height"][1:]
    mean_curvatures = np.zeros_like(h)
    for i in range(h.shape[0]):
        mean_curvatures[i] = util.mean_curvature(h[i], mc.x_step)

    binned_mean_curvatures = np.zeros(
        (sets_to_consider, mean_curvatures.shape[1], mean_curvatures.shape[2])
    )

    gro = util.analysis_path / f"{sim}/po4_only.gro"
    traj = util.analysis_path / f"{sim}/po4_all.xtc"

    u = MDAnalysis.Universe(gro, str(traj), refresh_offsets=True)
    ag = determine_leaflets(u, po4_neighbor_sel)

    for k, query in queries.items():
        print(f"analyzing lipid {k}")

        if len(u.select_atoms(query)) == 0:
            continue

        binned_lipid_density = np.zeros(
            (sets_to_consider, mean_curvatures.shape[1], mean_curvatures.shape[2])
        )

        for i in range(sets_to_consider):
            if i == 0:
                binned_mean_curvatures[i] = np.mean(
                    mean_curvatures[-frames_to_average:],
                    axis=0,
                )
            else:
                binned_mean_curvatures[i] = np.mean(
                    mean_curvatures[
                        -frames_to_average * (i + 1) : -frames_to_average * i
                    ],
                    axis=0,
                )

            upper_positions = []
            lower_positions = []

            if i == 0:
                for ts in u.trajectory[-frames_to_average:]:
                    xy = wrap_and_sanitize(
                        ag["upper"].select_atoms(query).positions[:, 0:2], ts, mc
                    )
                    upper_positions.append(xy)
                    xy = wrap_and_sanitize(
                        ag["lower"].select_atoms(query).positions[:, 0:2], ts, mc
                    )
                    lower_positions.append(xy)
            else:
                for ts in u.trajectory[
                    -frames_to_average * (i + 1) : -frames_to_average * i
                ]:
                    xy = wrap_and_sanitize(
                        ag["upper"].select_atoms(query).positions[:, 0:2], ts, mc
                    )
                    upper_positions.append(xy)
                    xy = wrap_and_sanitize(
                        ag["lower"].select_atoms(query).positions[:, 0:2], ts, mc
                    )
                    lower_positions.append(xy)
            # Convert to numpy array
            upper_positions = np.vstack(upper_positions)
            lower_positions = np.vstack(lower_positions)

            binned_lipid_density[i], xe, ye = np.histogram2d(
                upper_positions[:, 0],
                upper_positions[:, 1],
                bins=mc.n_x_bins,
                range=[
                    [mc.x_range[0] - mc.x_step / 2, mc.x_range[1] - mc.x_step / 2],
                    [mc.y_range[0] - mc.x_step / 2, mc.y_range[1] - mc.x_step / 2],
                ],
                density=True,
            )

        binned_mean_curvatures -= np.mean(binned_mean_curvatures)
        binned_lipid_density -= np.mean(binned_lipid_density)

        ## COMPUTE 2D SPATIAL CORRELATION
        acr = []
        for i in tqdm(range(sets_to_consider)):
            acr.append(
                signal.correlate2d(
                    binned_mean_curvatures[i],
                    binned_lipid_density[i],
                    boundary="wrap",
                )
                / binned_mean_curvatures.shape[1] ** 2
                / (np.std(binned_mean_curvatures[i]) * np.std(binned_lipid_density[i]))
            )

        acr = np.array(acr)
        acr = np.mean(acr, axis=0)

        for style, style_ext in plot_styles:
            with plt.style.context(style):
                if style_ext:
                    ecolor = "white"
                else:
                    ecolor = "black"

                shape_size = xe[-1] / 10
                fig, ax = plt.subplots(1, 1, figsize=(3, 3))  # sharex=True,
                im = ax.imshow(
                    acr,
                    vmin=-0.3,
                    vmax=0.3,
                    extent=[-shape_size, shape_size, -shape_size, shape_size],
                    origin="lower",
                )

                fig.colorbar(im, ax=ax)
                ax.set_ylabel("Y (nm)")
                ax.set_xlabel("X (nm)")

                limits = (-20, 20)
                ax.set_xlim(*limits)
                ax.set_ylim(*limits)

                if sim == "1_vbt":
                    fig.suptitle(f"1_vbt {k}")
                else:
                    fig.suptitle(f"sim {util.sim_to_final_index[int(sim)]} {k}")

                fig.tight_layout()

                if sim == "1_vbt":
                    save_fig(fig, curr_fig_path / f"1_vbt_correlation_{k}{style_ext}")
                else:
                    save_fig(
                        fig,
                        curr_fig_path
                        / f"{util.sim_to_final_index[int(sim)]}_correlation_{k}{style_ext}",
                    )

                if show_figs:
                    plt.show()

                fig.clear()
                plt.close(fig)


# %%
# show_figs = True
# curr_fig_path = Path("Figures/lipid_pearsons_scatter")
# curr_fig_path.mkdir(parents=True, exist_ok=True)


# frames_to_average = 5
# sets_to_consider = 200


# # for sim in np.concatenate((util.simulations, ["1_vbt"])):
# for sim in [1]:
#     print(f"sim {sim}")
#     with open(
#         util.analysis_path / f"{sim}/membrane_curvature_2nm.pickle", "rb"
#     ) as handle:
#         mc = pickle.load(handle)

#     h = mc.results["height"][1:]
#     mean_curvatures = np.zeros_like(h)
#     for i in range(h.shape[0]):
#         mean_curvatures[i] = util.mean_curvature(h[i], mc.x_step)

#     binned_mean_curvatures = np.zeros(
#         (sets_to_consider, mean_curvatures.shape[1], mean_curvatures.shape[2])
#     )

#     gro = util.analysis_path / f"{sim}/po4_only.gro"
#     traj = util.analysis_path / f"{sim}/po4_all.xtc"

#     u = MDAnalysis.Universe(gro, str(traj), refresh_offsets=True)
#     ag = determine_leaflets(u, po4_neighbor_sel)

#     for k, query in queries.items():
#         print(f"analyzing lipid {k}")

#         if len(u.select_atoms(query)) == 0:
#             continue

#         binned_lipid_density = np.zeros(
#             (sets_to_consider, mean_curvatures.shape[1], mean_curvatures.shape[2])
#         )

#         for i in range(sets_to_consider):
#             if i == 0:
#                 binned_mean_curvatures[i] = np.mean(
#                     mean_curvatures[-frames_to_average:],
#                     axis=0,
#                 )
#             else:
#                 binned_mean_curvatures[i] = np.mean(
#                     mean_curvatures[
#                         -frames_to_average * (i + 1) : -frames_to_average * i
#                     ],
#                     axis=0,
#                 )

#             upper_positions = []
#             lower_positions = []

#             if i == 0:
#                 for ts in u.trajectory[-frames_to_average:]:
#                     xy = ag["upper"].select_atoms(query).positions[:, 0:2]
#                     xy = np.where(xy > ts.dimensions[0]-mc.x_step, xy-ts.dimensions[0], xy)
#                     upper_positions.append(
#                         xy    
#                     )
#                     xy = ag["lower"].select_atoms(query).positions[:, 0:2]
#                     xy = np.where(xy > ts.dimensions[0]-mc.x_step, xy-ts.dimensions[0], xy)
#                     lower_positions.append(
#                         xy
#                     )
#             else:
#                 for ts in u.trajectory[
#                     -frames_to_average * (i + 1) : -frames_to_average * i
#                 ]:
#                     xy = ag["upper"].select_atoms(query).positions[:, 0:2]
#                     xy = np.where(xy > ts.dimensions[0]-mc.x_step, xy-ts.dimensions[0], xy)
#                     upper_positions.append(
#                         xy    
#                     )
#                     xy = ag["lower"].select_atoms(query).positions[:, 0:2]
#                     xy = np.where(xy > ts.dimensions[0]-mc.x_step, xy-ts.dimensions[0], xy)
#                     lower_positions.append(
#                         xy
#                     )
#             # Convert to numpy array
#             upper_positions = np.vstack(upper_positions)
#             lower_positions = np.vstack(lower_positions)

#             binned_lipid_density[i], xe, ye = np.histogram2d(
#                 upper_positions[:, 0],
#                 upper_positions[:, 1],
#                 bins=mc.n_x_bins,
#                 range=[[(mc.x_range[0])-10, mc.x_range[1]-10], [mc.y_range[0]-10, mc.y_range[1]-10]],
#                 density=True,
#             )

#         binned_mean_curvatures -= np.mean(binned_mean_curvatures)
#         binned_lipid_density -= np.mean(binned_lipid_density)

#         # ## COMPUTE 2D SPATIAL CORRELATION
#         # acr = []
#         # for i in tqdm(range(sets_to_consider)):
#         #     acr.append(
#         #         signal.correlate2d(
#         #             binned_mean_curvatures[i],
#         #             binned_lipid_density[i],
#         #             boundary="wrap",
#         #         )
#         #         / binned_mean_curvatures.shape[1] ** 2
#         #         / (np.std(binned_mean_curvatures[i]) * np.std(binned_lipid_density[i]))
#         #     )

#         # acr = np.array(acr)
#         # acr = np.mean(acr, axis=0)

#         # for style, style_ext in plot_styles:
#         #     with plt.style.context(style):
#         #         if style_ext:
#         #             ecolor = "white"
#         #         else:
#         #             ecolor = "black"

#         for direction in [-2, -1, 0, 1, 2]:
#             fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(9, 3))  # sharex=True,

#             ax1.plot(
#                 binned_mean_curvatures[50].ravel(),
#                 np.roll(binned_lipid_density[50], (direction, 0), axis=(1, 0)).ravel(),
#                 ".",
#             )

#             ax2.imshow(binned_mean_curvatures[50])
#             ax3.imshow(np.roll(binned_lipid_density[50], (direction, 0), axis=(1, 0)))

#             print(
#                 direction,
#                 stats.pearsonr(
#                     binned_mean_curvatures[50].ravel(),
#                     np.roll(
#                         binned_lipid_density[50], (direction, 0), axis=(1, 0)
#                     ).ravel(),
#                 ),
#             )

#             # im = ax.imshow(
#             #     acr,
#             #     vmin=-0.25,
#             #     vmax=0.25,
#             #     extent=[-shape_size, shape_size, -shape_size, shape_size],
#             #     origin="lower",
#             # )

#             # fig.colorbar(im, ax=ax)
#             # ax.set_ylabel("Y (nm)")
#             # ax.set_xlabel("X (nm)")

#             # limits = (-20, 20)
#             # ax.set_xlim(*limits)
#             # ax.set_ylim(*limits)

#             if sim == "1_vbt":
#                 fig.suptitle(f"1_vbt {k}")
#             else:
#                 fig.suptitle(f"sim {util.sim_to_final_index[int(sim)]} {k}")

#             fig.tight_layout()

#             if sim == "1_vbt":
#                 save_fig(fig, curr_fig_path / f"1_vbt_correlation_{k}{style_ext}")
#             else:
#                 save_fig(
#                     fig,
#                     curr_fig_path
#                     / f"{util.sim_to_final_index[int(sim)]}_correlation_{k}{style_ext}",
#                 )

#             if show_figs:
#                 plt.show()

#             fig.clear()
#             plt.close(fig)


# %%
