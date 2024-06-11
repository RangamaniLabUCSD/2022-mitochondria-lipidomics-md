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

from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

import util
from plot_helper import *


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

plot_styles = [
    ("./white_background_ctl.mplstyle", ""),
    ("./dark_background_ctl.mplstyle", "_dark"),
]


# %%
# Location to save the final data
curvature_correlation_fd = util.analysis_path / "gaussian_curvature_correlation.pickle"

if curvature_correlation_fd.exists():
    with open(curvature_correlation_fd, "rb") as handle:
        curvature_correlation_data = pickle.load(handle)
else:
    raise RuntimeError("Curvature correlation cache is missing")


# %%
show_figs = False
curr_fig_path = Path("Figures/gaussian_curvature_histograms")
curr_fig_path.mkdir(parents=True, exist_ok=True)

for sim in np.concatenate((util.simulations, ["1_vbt"])):
    ahs = curvature_correlation_data[sim]["all"]

    for lipid, query in queries.items():
        if lipid not in curvature_correlation_data[sim]:
            continue

        hs = curvature_correlation_data[sim][lipid]

        bin_range = [-0.02, 0.02]

        for style, style_ext in plot_styles:
            with plt.style.context(style):
                if style_ext:
                    ecolor = "white"
                else:
                    ecolor = "black"
                fig, ax = plt.subplots(1, 1, figsize=(3, 3))  # sharex=True,

                hsn, hs_bins, _ = ax.hist(
                    hs,
                    bins=101,
                    range=bin_range,
                    density=True,
                    histtype="step",
                    color="r",
                )

                asn, as_bins, _ = ax.hist(
                    ahs,
                    bins=101,
                    range=bin_range,
                    density=True,
                    # color="",
                    alpha=0.7,
                )

                print(f"{sim} Overall mean: {np.mean(ahs)}; {lipid} mean {np.mean(hs)}")

                # ax.hist(
                #     np.ravel(mean),
                #     bins=101,
                #     range=bin_range,
                #     density=True,
                #     color="k",
                #     alpha=0.7,
                # )

                ax.axvline(0, color=ecolor, linestyle="--", linewidth=1)

                ax.set_xlabel(r"Guassian curvature (1/nm^2)")
                ax.set_ylabel(r"Density")

                if sim == "1_vbt":
                    ax.set_title(f"1_vbt {lipid}")
                else:
                    ax.set_title(f"{util.sim_to_final_index[int(sim)]} {lipid}")
                ax.set_xlim(bin_range)

                # ax.legend(loc="upper right")

                # # Shrink current axis by 20%
                # box = ax.get_position()
                # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                # # Put a legend to the right of the current axis
                # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

                fig.tight_layout()

                if sim == "1_vbt":
                    save_fig(fig, curr_fig_path / f"1_vbt_{lipid}{style_ext}")
                else:
                    save_fig(
                        fig,
                        curr_fig_path
                        / f"{util.sim_to_final_index[int(sim)]}_{lipid}{style_ext}",
                    )

                if show_figs:
                    plt.show()

                fig.clear()
                plt.close(fig)


# %%
show_figs = True
curr_fig_path = Path("Figures/gaussian_curvature_histograms_diff")
curr_fig_path.mkdir(parents=True, exist_ok=True)

for sim in np.concatenate((util.simulations, ["1_vbt"])):
    ahs = curvature_correlation_data[sim]["all"]

    for lipid, query in queries.items():
        if lipid not in curvature_correlation_data[sim]:
            continue

        hs = curvature_correlation_data[sim][lipid]
        bin_range = [-0.02, 0.02]

        hsn, hs_bins = np.histogram(hs, bins=101, range=tuple(bin_range), density=True)
        asn, as_bins = np.histogram(ahs, bins=101, range=tuple(bin_range), density=True)
        print(f"{sim} Overall mean: {np.mean(ahs)}; {lipid} mean {np.mean(hs)}")

        for style, style_ext in plot_styles:
            with plt.style.context(style):
                if style_ext:
                    ecolor = "white"
                else:
                    ecolor = "black"
                fig, ax = plt.subplots(1, 1, figsize=(3, 3))  # sharex=True,

                # Lipid specific minus all
                ax.bar(hs_bins[:-1], hsn - asn, width=hs_bins[1] - hs_bins[0])
                ax.axvline(0, color=ecolor, linestyle="--", linewidth=1)

                ax.set_xlabel(r"Gaussian curvature (1/nm^2)")
                ax.set_ylabel(r"Density")

                if sim == "1_vbt":
                    ax.set_title(f"1_vbt {lipid}")
                else:
                    ax.set_title(f"{util.sim_to_final_index[int(sim)]} {lipid}")
                ax.set_xlim(bin_range)
                # ax.set_ylim(-0.5, 0.5)

                fig.tight_layout()

                if sim == "1_vbt":
                    save_fig(fig, curr_fig_path / f"1_vbt_{lipid}{style_ext}")
                else:
                    save_fig(
                        fig,
                        curr_fig_path
                        / f"{util.sim_to_final_index[int(sim)]}_{lipid}{style_ext}",
                    )

                if show_figs:
                    plt.show()

                fig.clear()
                plt.close(fig)


# %%
curvature_first_moment = {}
curvature_zero_moment = {}
curvature_second_moment = {}

for sim in np.concatenate((util.simulations, ["1_vbt"])):
    curvature_zero_moment[sim] = {}
    curvature_first_moment[sim] = {}
    curvature_second_moment[sim] = {}
    ahs = curvature_correlation_data[sim]["all"]

    for lipid, query in queries.items():
        if lipid not in curvature_correlation_data[sim]:
            continue

        hs = curvature_correlation_data[sim][lipid]
        bin_range = [-0.02, 0.02]
        hsn, hs_bins = np.histogram(hs, bins=101, range=bin_range, density=True)
        asn, as_bins = np.histogram(ahs, bins=101, range=bin_range, density=True)

        # Lipid specific minus all
        diff = hsn - asn
        bin_centers = ((hs_bins + np.roll(hs_bins, -1)) / 2)[:-1]

        moment = np.sum(diff * bin_centers) * (bin_centers[1] - bin_centers[0])
        second_moment = np.sum(diff * bin_centers * bin_centers) * (
            bin_centers[1] - bin_centers[0]
        )
        curvature_zero_moment[sim][lipid] = np.sum(diff) * (
            bin_centers[1] - bin_centers[0]
        )

        curvature_first_moment[sim][lipid] = moment
        curvature_second_moment[sim][lipid] = second_moment
        if sim != "1_vbt":
            print(
                f"{sim} {util.sim_to_final_index[int(sim)]} moment: {moment}; Overall mean: {np.mean(ahs)}; {lipid} mean {np.mean(hs)}"
            )
        else:
            print(
                f"{sim} moment: {moment}; Overall mean: {np.mean(ahs)}; {lipid} mean {np.mean(hs)}"
            )


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

# palette = [lighten_color(p[i], j) for i, j in [(7, 1),(7, light_factor),(8,1),(8,light_factor),(0,1),(0,light_factor),(2,1),(2,light_factor)]]

# sns.palplot(palette)
# sns.palplot(p)


# %%
show_figs = True
curr_fig_path = Path("Figures/gaussian_delta_analysis")
curr_fig_path.mkdir(parents=True, exist_ok=True)

color_index = {"CDL": 1, "PC": 7, "PG": 2, "PE": 8}

vals = []
colors = []
for sim in range(1, 16):
    for lipid in queries.keys():
        if lipid not in curvature_first_moment[str(util.remapping_dict[sim])]:
            continue
        vals.append(curvature_first_moment[str(util.remapping_dict[sim])][lipid])
        colors.append(p[color_index[lipid]])

# print(vals)
# print(colors)


for style, style_ext in plot_styles:
    with plt.style.context(style):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))  # sharex=True,

        ax.bar(range(len(vals)), vals, color=colors)

        ax.set_xticks(np.arange(1, len(vals), 3))
        ax.set_xticklabels(np.arange(1, 16))

        patches = [
            mpatches.Patch(color=p[1], label="CDL"),
            mpatches.Patch(color=p[7], label="PC"),
            mpatches.Patch(color=p[2], label="PG"),
            mpatches.Patch(color=p[8], label="PE"),
        ]

        box = ax.get_position()
        ax.legend(
            handles=patches, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.13)
        )

        ax.set_ylabel("First moment of Δ")
        ax.set_xlabel("System")

        fig.tight_layout()

        save_fig(fig, curr_fig_path / f"first_moment_1-9{style_ext}")

        if show_figs:
            plt.show()

        fig.clear()
        plt.close(fig)


# %%
show_figs = True
curr_fig_path = Path("Figures/gaussian_delta_analysis")
curr_fig_path.mkdir(parents=True, exist_ok=True)

color_index = {"CDL": 1, "PC": 7, "PG": 2, "PE": 8}

vals = []
colors = []
for sim in range(1, 16):
    for lipid in queries.keys():
        if lipid not in curvature_zero_moment[str(util.remapping_dict[sim])]:
            continue
        vals.append(curvature_zero_moment[str(util.remapping_dict[sim])][lipid])
        colors.append(p[color_index[lipid]])

# print(vals)
# print(colors)


for style, style_ext in plot_styles:
    with plt.style.context(style):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))  # sharex=True,

        ax.bar(range(len(vals)), vals, color=colors)

        ax.set_xticks(np.arange(1, len(vals), 3))
        ax.set_xticklabels(np.arange(1, 16))

        patches = [
            mpatches.Patch(color=p[1], label="CDL"),
            mpatches.Patch(color=p[7], label="PC"),
            mpatches.Patch(color=p[2], label="PG"),
            mpatches.Patch(color=p[8], label="PE"),
        ]

        box = ax.get_position()
        ax.legend(
            handles=patches, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.13)
        )

        ax.set_ylabel("Zeroth moment of Δ")
        ax.set_xlabel("System")

        fig.tight_layout()

        save_fig(fig, curr_fig_path / f"zero_moment_1-9{style_ext}")

        if show_figs:
            plt.show()

        fig.clear()
        plt.close(fig)


# %%
show_figs = True
curr_fig_path = Path("Figures/gaussian_delta_analysis")
curr_fig_path.mkdir(parents=True, exist_ok=True)

color_index = {"CDL": 1, "PC": 7, "PG": 2, "PE": 8}

vals = []
colors = []
for sim in range(1, 16):
    for lipid in queries.keys():
        if lipid not in curvature_second_moment[str(util.remapping_dict[sim])]:
            continue
        vals.append(curvature_second_moment[str(util.remapping_dict[sim])][lipid])
        colors.append(p[color_index[lipid]])

# print(vals)
# print(colors)


for style, style_ext in plot_styles:
    with plt.style.context(style):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))  # sharex=True,

        ax.bar(range(len(vals)), vals, color=colors)

        ax.set_xticks(np.arange(1, len(vals), 3))
        ax.set_xticklabels(np.arange(1, 16))

        patches = [
            mpatches.Patch(color=p[1], label="CDL"),
            mpatches.Patch(color=p[7], label="PC"),
            mpatches.Patch(color=p[2], label="PG"),
            mpatches.Patch(color=p[8], label="PE"),
        ]

        box = ax.get_position()
        ax.legend(
            handles=patches, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.13)
        )

        ax.set_ylabel("Second moment of Δ")
        ax.set_xlabel("System")

        fig.tight_layout()

        save_fig(fig, curr_fig_path / f"second_moment_1-9{style_ext}")

        if show_figs:
            plt.show()

        fig.clear()
        plt.close(fig)


# %%
