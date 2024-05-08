# %% [markdown]
# # Analysis of material properties of mitochondrial membranes

# %% [markdown]
# ## Loading and setup

# %%
%load_ext autoreload


# %%
%matplotlib inline
%autoreload 1
import pickle
import numpy as np
from functools import partial
import MDAnalysis

from pathlib import Path

import matplotlib.pyplot as plt
import numpy.typing as npt

import pandas as pd

from scipy import integrate, interpolate, stats
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

%aimport util
from plot_helper import *


# %%
plot_styles = [
    ("./white_background_ctl.mplstyle", ""),
    ("./dark_background_ctl.mplstyle", "_dark"),
]


# %%
def radial_averaging(power2D, mc, min_bin=0.001, max_bin=1, bin_width=0.001):
    """
    Radially average the power spectrum to obtain values. Notably the natural freqeuncy unit
    of this function is A^-1.

    Args:
        power2D (numpy.array((N,N))): Power spectrum
        mc (_type_): Membrane curvature object with metadata
        min_bin (float, optional): Minimum bin value. Defaults to 0.001.
        max_bin (int, optional): Maximum bin value. Defaults to 1.
        bin_width (float, optional): Bin width. Defaults to 0.001.

    Returns:
        tuple: Binned power spectra
    """
    x, y = np.meshgrid(mc["qx"], mc["qy"])  # A^-1
    r = np.sqrt(x**2 + y**2)
    bins = np.arange(min_bin, max_bin, bin_width)

    digitized = np.digitize(r, bins)
    bc = np.array(
        [
            r[digitized == i].mean() if np.count_nonzero(digitized == i) else np.NAN
            for i in range(1, len(bins))
        ]
    )
    bm = np.array(
        [
            power2D[digitized == i].mean()
            if np.count_nonzero(digitized == i)
            else np.NAN
            for i in range(1, len(bins))
        ]
    )

    bin_centers = bc[np.isfinite(bm)]
    bin_means = bm[np.isfinite(bm)]

    return np.column_stack((bin_centers, bin_means, bin_centers**4 * bin_means))


def radial_averaging_series(power2D, mc, min_bin=0.001, max_bin=1, bin_width=0.001):
    """
    Perform radial averaging over multiple frames in a time series.

    Radially average the power spectrum to obtain values. Notably the natural freqeuncy unit
    of this function is A^-1.

    Args:
        power2D (numpy.array((M,N,N))): Power spectrum
        mc (_type_): Membrane curvature object with metadata
        min_bin (float, optional): Minimum bin value. Defaults to 0.001.
        max_bin (int, optional): Maximum bin value. Defaults to 1.
        bin_width (float, optional): Bin width. Defaults to 0.001.

    Returns:
        tuple: Binned power spectra
    """

    if not len(power2D.shape) == 3:
        raise RuntimeError("Expected time series of 2D power")

    x, y = np.meshgrid(mc["qx"], mc["qy"])  # A^-1
    r = np.sqrt(x**2 + y**2)
    bins = np.arange(min_bin, max_bin, bin_width)

    digitized = np.digitize(r, bins)
    bc = np.array(
        [
            r[digitized == i].mean() if np.count_nonzero(digitized == i) else np.NAN
            for i in range(1, len(bins))
        ]
    )

    first_iter = True

    spectra = None

    for i, frame in tqdm(enumerate(power2D), total=len(power2D)):
        bm = np.array(
            [
                frame[digitized == i].mean()
                if np.count_nonzero(digitized == i)
                else np.NAN
                for i in range(1, len(bins))
            ]
        )

        if i == 0:
            bin_centers = bc[np.isfinite(bm)]
            bin_means = bm[np.isfinite(bm)]
            spectra = np.zeros((power2D.shape[0], len(bin_means)))
            spectra[i] = bin_means
        else:
            spectra[i] = bm[np.isfinite(bm)]
    return (bin_centers, spectra)


# def radial_averaging_nm(power2D, mc, min_bin=0.1, max_bin=10, bin_width=0.1):
#     x, y = np.meshgrid(mc.qx * 10, mc.qy * 10)  # convert to nm^-1
#     r = np.sqrt(x**2 + y**2)
#     bins = np.arange(min_bin, max_bin, bin_width)

#     digitized = np.digitize(r, bins)
#     bc = np.array(
#         [
#             r[digitized == i].mean() if np.count_nonzero(digitized == i) else np.NAN
#             for i in range(1, len(bins))
#         ]
#     )
#     bm = np.array(
#         [
#             power2D[digitized == i].mean()
#             if np.count_nonzero(digitized == i)
#             else np.NAN
#             for i in range(1, len(bins))
#         ]
#     )

#     bin_centers = bc[np.isfinite(bm)]
#     bin_means = bm[np.isfinite(bm)]

#     return np.column_stack((bin_centers, bin_means, bin_centers**4 * bin_means))


def count_residues(u):
    count_dict = {}
    for residue in u.residues:
        if residue.resname not in count_dict:
            count_dict[residue.resname] = 1
        else:
            count_dict[residue.resname] += 1
    return count_dict


# %% [markdown]
# ### System information

# %%
def get_compositions(sim):
    top = util.sim_path / str(sim) / "system.top"

    raw_composition = {}
    with open(top, "r") as fd:
        molecules_flag = False
        for line in fd:
            if molecules_flag:
                line = line.split(";")[0]
                if line:
                    r, n = line.split()
                    if r in raw_composition:
                        raw_composition[r] += int(n)
                    else:
                        raw_composition[r] = int(n)
            else:
                if "[ molecules ]" in line:
                    molecules_flag = True

    total_lipids = 0
    for lipid in util.lipid_names:
        if lipid in raw_composition:
            total_lipids += raw_composition[lipid]

    normed_composition = {}
    s = ""
    for lipid in util.lipid_names:
        if lipid in raw_composition:
            s += f"{lipid}: {raw_composition[lipid]/total_lipids:0.2f}; "
            normed_composition[lipid] = raw_composition[lipid] / total_lipids
        else:
            s += f"{lipid}: {0:0.2f}; "
            normed_composition[lipid] = 0
    print(util.sim_to_final_index[sim], "total lipids", total_lipids)
    return sim, raw_composition, normed_composition, s


result = map(get_compositions, util.simulations)

compositions = {}
for sim, raw, normed, s in result:
    print(f"System {util.sim_to_final_index[sim]}: {s}")
    print(f"    {raw}")
    compositions[sim, "raw_composition"] = raw
    compositions[sim, "normed_composition"] = normed


# %%
order = [
    "CDL1",
    "CDL2",
    "DOPG",
    "POPG",
    "DOPC",
    "POPC",
    "DOPE",
    "POPE",
    "W",
    "NA",
    "CL",
]


for sim in util.remapping_order:
    composition = compositions[sim, "raw_composition"]
    temp_str = f"System {util.sim_to_final_index[sim]} & "
    for moltype in order:
        if moltype in composition:
            temp_str += f"& {composition[moltype]} "
        else:
            temp_str += "& 0 "
    temp_str += "\\\\"

    print(temp_str)


# %%
# def get_compositions(sim):
#     top = util.sim_path / str(sim) / "system.top"


#     raw_composition = {}
#     with open(top, 'r') as fd:
#         molecules_flag = False
#         for line in fd:
#             if molecules_flag:
#                 line = line.split(";")[0]
#                 if line:
#                     r, n = line.split()
#                     if r in raw_composition:
#                         raw_composition[r] += int(n)
#                     else:
#                         raw_composition[r] = int(n)
#             else:
#                 if "[ molecules ]" in line:
#                     molecules_flag = True

#     total_lipids = 0
#     for lipid in util.lipid_names:
#         if lipid in raw_composition:
#             total_lipids += raw_composition[lipid]

#     normed_composition = {}
#     s = ""
#     for lipid in util.lipid_names:
#         if lipid in raw_composition:
#             s += f"{lipid}: {raw_composition[lipid]/total_lipids:0.2f}; "
#             normed_composition[lipid] = raw_composition[lipid] / total_lipids
#         else:
#             s += f"{lipid}: {0:0.2f}; "
#             normed_composition[lipid] = 0
#     # print(util.sim_to_final_index[sim], "total lipids", total_lipids)
#     return sim, raw_composition, normed_composition, s


# result = map(get_compositions, util.simulations)

# import pandas as pd

# compositions = {}
# lipid_names = ["POPC", "DOPC", "POPE", "DOPE", "CDL1", "CDL2", "POPG", "DOPG"]
# print(lipid_names)
# for sim, raw, normed, s in result:
#     tmp = f"System {util.sim_to_final_index[sim]}, "
#     for lipid in lipid_names:
#         if lipid in raw:
#             tmp += f"{raw[lipid]}, "
#         else:
#             tmp += f"0, "
#     print(tmp)
#     # compositions[sim, "raw_composition"] = raw
#     # compositions[sim, "normed_composition"] = normed


# %% [markdown]
# ## Defining Statistical Inefficiency

# %% [markdown]
# Given a sequence of measurements $A_i$ sampled from a timeseries, we must investigate the degree of correlation to estimate the sampling error. We estimate the error by quantifying the statistical inefficiency.
# 
# We start by computing the block averaged values, $\langle A\rangle_b$ over a range of block lengths $t_b$,
# $$\langle A\rangle_b = \frac{1}{t_b} \sum_{i=1}^{t_b} A_i.$$
# 
# As the number of steps increases, we expect that the block averages become uncorrelated. The variance of block averages $\sigma^2(\langle A\rangle_b)$,
# $$\sigma^2(\langle A\rangle_b) = \frac{1}{n_b}\sum_{b=1}^{n_b} (\langle A\rangle_b - \langle A_i\rangle)^2,$$
# becomes inversely proportional to $t_b$ as the block averages become uncorrelated.
# 
# At the uncorrelated limit, the statistical inefficiency is given by,
# $$ s = \lim_{t_b\rightarrow \infty} \frac{t_b \sigma^2(\langle A\rangle_b)}{\sigma^2(A)}.$$
# 
# The 'true' standard deviation of the average value is then related to the traditional standard deviation by,
# $$\sigma_{\langle A\rangle} \approx \sigma \sqrt{\frac{s}{M}}.$$

# %% [markdown]
# ## Setting up parametric error analysis

# %% [markdown]
# Given a sequence of evenly space measurements ${X_1, X_2,\ldots, X_T}$ along a trajectory, the sample mean $m_X$ and sample variance $s^2_X$ is given by
# 
# $$m_X = \frac{1}{T} \sum_{i=1}^{T}X_i,$$
# $$s^2_X = \frac{1}{T-1}\sum_{i=1}^{T}(X_i - m_X)^2.$$
# 
# The error of the mean can be estimated using $\delta X = s_X / \sqrt{T}$ if the data are uncorrelated. Since the measurements are sampled from a dynamical trajectory, there is no guarantee that there is no correlation.
# 

# %% [markdown]
# ## Kc Bending Modulus

# %%
kc_low_q = 0.4 / 10  # A^-1


def fit_kc_from_power(
    power2D, mc=None, threshold=0.03, min_bin=0.001, max_bin=1, bin_width=0.001
):
    spectra = radial_averaging(
        power2D, mc, min_bin=min_bin, max_bin=max_bin, bin_width=bin_width
    )
    mask = spectra[:, 0] < threshold
    spectra_cut = spectra[mask, :]

    return 1.0 / spectra_cut[:, 2].mean()


# %% [markdown]
# ### Block analysis of Fourier modes

# %%
# Override and recompute even if spectra pickle exists
spectra_compute_override = False

suffix = "_2nm"


spectra_fd = util.analysis_path / ("spectra" + suffix + ".pickle")
if spectra_fd.exists() and not spectra_compute_override:
    # LOAD SPECTRA PICKLE
    with open(spectra_fd, "rb") as handle:
        spectra = pickle.load(handle)
    print("Loaded spectra from cache!")
else:
    # Generate this using non_interactive_radial averaging
    with open("mc_noobject" + suffix + ".pickle", "rb") as handle:
        mc = pickle.load(handle)

    def compute_spectra(sim):
        return sim, radial_averaging_series(
            mc[sim]["height_power_spectrum"],
            mc[sim],
            min_bin=0.001,
            max_bin=1,
            bin_width=0.001,
        )

    spectra = dict(map(compute_spectra, util.simulations))

    # WRITE SPECTRA TO PICKLE
    with open(spectra_fd, "wb") as handle:
        pickle.dump(spectra, handle, protocol=pickle.HIGHEST_PROTOCOL)


# POPULATE q^4*h_q
qfour_spectra = {}
for sim in util.simulations:
    qfour_spectra[sim] = np.power(spectra[sim][0], 4) * spectra[sim][1]


# %% [markdown]
# #### Timeseries of wavenumbers

# %%
show_figs = False
curr_fig_path = Path("Figures/power_timeseries")
curr_fig_path.mkdir(parents=True, exist_ok=True)

## PLOT HEIGHT POWER TIMESERIES
for sim in util.simulations:
    for style, style_ext in plot_styles:
        with plt.style.context(style):
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))  # sharex=True,

            for i in range(0, 4):
                ax.plot(
                    range(len(spectra[sim][1][:, i])),
                    spectra[sim][1][:, i],
                    linewidth=NORMAL_LINE,
                    label=f"q{i}",
                )

                # plot q^4*h_q instead
                # ax.plot(
                #     range(len(spectra[sim][1][:, i])),
                #     np.power(spectra[sim][0][i],4)*spectra[sim][1][:, i],
                #     linewidth=NORMAL_LINE,
                #     label=f"q{i}",
                # )

            ax.set_xlabel(r"Frame")
            ax.set_ylabel(r"Power Spectrum $\langle|h_q|\rangle^2$ (nm$^{4}$)")
            ax.set_title(f"{util.sim_to_final_index[sim]}:{util.system_names[sim]}")

            ax.legend(loc="upper right")

            fig.tight_layout()
            save_fig(
                fig,
                curr_fig_path
                / f"{util.sim_to_final_index[sim]}_power_timeseries{style_ext}",
            )

            if show_figs:
                plt.show()

            fig.clear()
            plt.close(fig)


# %% [markdown]
# #### Statistical inefficiency of Fourier amplitudes

# %%
show_figs = False
curr_fig_path = Path("Figures/amplitude_si")
curr_fig_path.mkdir(parents=True, exist_ok=True)

## COMPUTE STATISTICAL INEFFICIENCY OF WAVE NUMBERS UP TO max_q
max_q = 4
discards = np.arange(0, 60, 10)
blocks = np.arange(1, 2**8 + 1, 1)

cmap = mpl.cm.get_cmap("viridis")

for sim in util.simulations:
    for q in range(0, max_q):
        for style, style_ext in plot_styles:
            with plt.style.context(style):
                fig, ax = plt.subplots(1, 1, figsize=(3, 3))  # sharex=True,
                c = cmap(np.linspace(0, 1, len(discards)))

                _, _, si = util.statistical_inefficiency(
                    qfour_spectra[sim][:, q], blocks, discards
                )

                for d, discard in enumerate(discards):
                    ax.plot(
                        blocks,
                        si[d],
                        color=c[d],
                        linewidth=NORMAL_LINE,
                        label=f"{100-discard}%",
                    )

                ax.set_xlabel(r"Blocks")
                ax.set_ylabel(r"Statistical inefficiency")
                ax.set_title(
                    f"{util.sim_to_final_index[sim]}:{util.system_names[sim]}, q4Hq{q}"
                )
                ax.legend(loc="upper left")
                fig.tight_layout()

                # # Shrink current axis by 20%
                # box = ax.get_position()
                # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                # # Put a legend to the right of the current axis
                # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

                save_fig(
                    fig,
                    curr_fig_path
                    / f"{util.sim_to_final_index[sim]}_q4Hq{q}{style_ext}",
                )

                if show_figs:
                    plt.show()
                fig.clear()
                plt.close(fig)


# %% [markdown]
# Conclude that the wavenumbers appear to equilibrate rapidly and we can keep the majority of the trajectory. Henceforth we will discard 10% from the beginning.

# %% [markdown]
# #### Block averaging of amplitudes

# %% [markdown]
# The correlation time of the squared standard error of the mean should follow such a trend:
# $$\frac{\delta X^2_b}{\delta X^2_1} = \frac{1+c_t}{1-c_t} - \frac{2*c_t}{b} * \frac{1-c^b_t}{(1-c_t)^2}$$

# %%
def correlation_time_sqrt(b, tau):
    ct = np.exp(-1 / tau)
    cb = np.exp(-b / tau)
    return np.sqrt((1 + ct) / (1 - ct) - (2 * ct) / b * (1 - cb) / np.power(1 - ct, 2))


def correlation_time(b, tau):
    ct = np.exp(-1 / tau)
    cb = np.exp(-b / tau)
    return (1 + ct) / (1 - ct) - (2 * ct) / b * (1 - cb) / np.power(1 - ct, 2)


# %%
# Discard first X% for all trajectories
discard = 10
max_q_dict = {}
blocks = np.arange(1, 2**9 + 1, 1)

block_var = {}
lp_block_sem = {}
block_mean = {}
for sim in util.simulations:
    max_q = sum(spectra[sim][0] < kc_low_q)

    max_q_dict[sim] = max_q

    block_mean[sim] = np.zeros((max_q, len(blocks)))
    block_var[sim] = np.zeros((max_q, len(blocks)))
    lp_block_sem[sim] = np.zeros((max_q, len(blocks)))

    low_q_data = qfour_spectra[sim][:, 0:max_q]
    # low_q_data = spectra[sim][1][:, 0:max_q]

    _, remainder = np.split(low_q_data, [int(discard / 100 * len(low_q_data))])

    block_mean[sim] = util.nd_block_average(
        remainder, axis=0, func=np.mean, blocks=blocks
    )
    block_var[sim] = util.nd_block_average(
        remainder, axis=0, func=partial(np.var, ddof=1), blocks=blocks
    )
    lp_block_sem[sim] = util.nd_block_average(
        remainder, axis=0, func=partial(stats.sem, ddof=1), blocks=blocks
    )


# %%
show_figs = False
curr_fig_path = Path("Figures/kc_block_error")
curr_fig_path.mkdir(parents=True, exist_ok=True)

corrected_mean_sem = {}

for sim in util.simulations:
    for style, style_ext in plot_styles:
        with plt.style.context(style):
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))  # sharex=True,

            corrected_mean_sem[sim] = np.empty((2, max_q_dict[sim]))

            for q in range(0, max_q_dict[sim]):
                # Mean with block size 1
                corrected_mean_sem[sim][0, q] = block_mean[sim][q][0]
                blocked_sem = lp_block_sem[sim][q]
                popt, pcov = curve_fit(
                    correlation_time_sqrt, blocks, blocked_sem / blocked_sem[0]
                )
                corrected_mean_sem[sim][1, q] = blocked_sem[0] * np.sqrt(2 * popt[0])

                ax.plot(
                    np.log2(blocks),
                    blocked_sem / blocked_sem[0],
                    linewidth=NORMAL_LINE,
                    label=f"q{q}",
                    color=sns.color_palette("colorblind")[q],
                    linestyle=":",
                )

                ax.plot(
                    np.log2(blocks),
                    [correlation_time_sqrt(block, popt) for block in blocks],
                    linewidth=NORMAL_LINE,
                    color=sns.color_palette("colorblind")[q],
                )

            ax.set_xlabel(r"$log_2$(block)")
            ax.set_ylabel(r"$\delta X_b/\delta X_1$")
            ax.set_title(f"{util.sim_to_final_index[sim]}:{util.system_names[sim]}")
            ax.legend(loc="upper left")

            fig.tight_layout()

            save_fig(fig, curr_fig_path / f"{util.sim_to_final_index[sim]}_block_error{style_ext}")

            if show_figs:
                plt.show()
            fig.clear()
            plt.close(fig)

np.save("corrected_mean_sem.npy", corrected_mean_sem)


# %%
show_figs = True
curr_fig_path = Path("Figures")
curr_fig_path.mkdir(parents=True, exist_ok=True)


cmap = mpl.cm.get_cmap("viridis")
# c = cmap(np.linspace(0, 1, len(util.simulations)))
c = cmap(np.linspace(0, 1, 21))

kc_mean_std = {}


for style, style_ext in plot_styles:
    with plt.style.context(style):
        # Bootstrap values
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        for i, sim in enumerate(range(1, 22)):
            sim = util.remapping_dict[sim]
            f_rvs = []

            # Construct random Gaussian process for each wavenumber
            for q in range(corrected_mean_sem[sim].shape[1]):
                # print(corrected_mean_sem[sim][0, i], corrected_mean_sem[sim][1][i])
                r = stats.norm(
                    loc=corrected_mean_sem[sim][0, q],
                    scale=corrected_mean_sem[sim][1][q],
                )
                f_rvs.append(r.rvs)
            # Run parametric bootstrap with random proceses
            boot = util.parametric_bootstrap(f_rvs, n_samples=50000)

            # Fit kcs to bootstrap samples
            kcs = 1.0 / np.mean(boot, axis=0) * 1.2

            ### SCALE 120% for VBT contribution...
            kc_mean_std[sim] = [np.mean(kcs), np.std(kcs)]

            print(sim, kc_mean_std[sim])

            if util.sim_to_final_index[sim] < 0:
                continue

            # Plot distribution of kce)
            ax.hist(
                kcs,
                bins=50,
                density=True,
                color=c[i],
                label=f"{util.sim_to_final_index[sim]}",
            )

        ax.set_xlabel(r"$K_c$ ($k_BT$)")
        ax.set_ylabel("Density")

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # ax.set_xlim([0,25])

        # Put a legend to the right of the current axis
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        save_fig(fig, curr_fig_path / f"kc_distributions{style_ext}")

        if show_figs:
            plt.show()
        fig.clear()
        plt.close(fig)


np.save("kc_mean_std.npy", kc_mean_std)

# %%
kc_mean_std = np.load("kc_mean_std.npy", allow_pickle=True).item()


# %%
show_figs = False
curr_fig_path = Path("Figures/height_spectra_kc")
curr_fig_path.mkdir(parents=True, exist_ok=True)

# # Discard first X% for all trajectories
discard = 10
max_q = 100

for i, sim in enumerate(util.simulations):
    low_q_data = qfour_spectra[sim][:, 0:max_q]
    # low_q_data = spectra[sim][1][:, 0:max_q]

    _, remainder = np.split(low_q_data, [int(discard / 100 * len(low_q_data))])

    # Convert to nm^-1
    q = spectra[sim][0][0:max_q] * 10
    mask = q < kc_low_q * 10

    # q^4*hq
    hq_mean = np.mean(remainder, axis=0)
    hq_std = np.std(remainder, axis=0)

    for style, style_ext in plot_styles:
        with plt.style.context(style):
            if style_ext:
                ecolor='white'
            else:
                ecolor='black'
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax.errorbar(
                q[mask],
                hq_mean[mask],
                yerr=hq_std[mask],
                fmt=".",
                markersize=3,
                elinewidth=THIN_LINE,
                ecolor="lightgray",
            )

            ax.errorbar(
                q[~mask],
                hq_mean[~mask],
                yerr=hq_std[~mask],
                color="dimgray",
                fmt=".",
                markersize=3,
                elinewidth=THIN_LINE,
                ecolor="lightgray",
            )

            ax.axhline(1 / kc_mean_std[sim][0], color="r")
            # ax.axvline(kc_low_q, color="k", linewidth=0.5, linestyle=":")

            ax.text(
                0.05,
                0.7,
                f"$K_c$ = {kc_mean_std[sim][0]:.1f} $\pm$ {kc_mean_std[sim][1]:.1f} $k_BT$",
                color="r",
                transform=ax.transAxes,
            )

            # ax.set_xlim(5e-2, 5)
            ax.set_ylim(0.0, 0.45)
            ax.set_xscale("log")

            ax.set_ylabel(r"$q^4 \times \mathrm{intensity}$ ($k_BT$)")
            ax.set_xlabel(r"$q$ (nm$^{-1}$)")

            ax.set_title(
                f"System {util.sim_to_final_index[sim]}: {util.system_names[sim]}"
            )
            fig.set_tight_layout(True)

            save_fig(
                fig,
                curr_fig_path
                / f"{util.sim_to_final_index[sim]}_height_spectra_kc{style_ext}",
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


pal = sns.color_palette("colorblind")
bar_props = [
    (1, 1, None),
    (3, 1, None),
    (2, 1.1, None),
    (2, 0.95, "///"),
]


# %%
show_figs = True
curr_fig_path = Path("Figures")
curr_fig_path.mkdir(parents=True, exist_ok=True)

for style, style_ext in plot_styles:
    with plt.style.context(style):
        if style_ext:
            ecolor = "white"
        else:
            ecolor = "black"
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        for i in range(1, 4):
            ax.barh(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                xerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[1],
                ecolor=ecolor,
                height=0.8,
            )

        for i in range(4, 7):
            ax.barh(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                xerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[3],
                ecolor=ecolor,
                height=0.8,
            )

        for i in range(7, 10):
            ax.barh(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                xerr=kc_mean_std[util.remapping_dict[i]][1],
                color=lighten_color(pal[2], 1.1),
                ecolor=ecolor,
                height=0.8,
            )

        for i in range(10, 12):
            ax.barh(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                xerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[1],
                height=0.8,
            )

        for i in range(12, 14):
            ax.barh(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                xerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[3],
                ecolor=ecolor,
                height=0.8,
            )

        for i in range(14, 16):
            ax.barh(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                xerr=kc_mean_std[util.remapping_dict[i]][1],
                color=lighten_color(pal[2], 1.1),
                ecolor=ecolor,
                height=0.8,
            )

        for i in range(16, 18):
            ax.barh(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                xerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[1],
                ecolor=ecolor,
                height=0.8,
            )

        for i in range(18, 20):
            ax.barh(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                xerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[3],
                ecolor=ecolor,
                height=0.8,
            )

        ax.barh(
            "20",
            kc_mean_std[util.remapping_dict[20]][0],
            xerr=kc_mean_std[util.remapping_dict[20]][1],
            color=pal[1],
            ecolor=ecolor,
            height=0.8,
        )

        ax.barh(
            "21",
            kc_mean_std[util.remapping_dict[21]][0],
            xerr=kc_mean_std[util.remapping_dict[21]][1],
            color=pal[3],
            ecolor=ecolor,
            height=0.8,
        )

        # ax.axhline(2.5, color="k")
        # ax.axhline(5.5, color="k")
        # ax.axhline(8.5, color="k")
        # ax.axhline(14.5, color="k")
        # ax.axhline(18.5, color="k")

        ax.set_xlabel(r"$K_c$ ($k_BT$)")
        ax.set_ylabel(r"System")

        ax.set_xlim(0, 25)
        ax.set_xticks(np.arange(0,25,5))
        # ax.set_xticklabels(
        #     ax.get_xticks(),
        # )

        # x_ticks_labels = [f"{sim}" for sim in range(1,22)]

        # Set number of ticks for x-axis
        # ax.set_xticks(range(21))
        # Set ticks labels for x-axis
        # ax.set_xticklabels(x_ticks_labels)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        fig.tight_layout()

        save_fig(fig, curr_fig_path / f"estimated_kcs{style_ext}")

        if show_figs:
            plt.show()

        fig.clear()
        plt.close(fig)


# %%
show_figs = True
curr_fig_path = Path("Figures")
curr_fig_path.mkdir(parents=True, exist_ok=True)

for style, style_ext in plot_styles:
    with plt.style.context(style):
        if style_ext:
            ecolor='white'
        else:
            ecolor='black'
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        for i in range(1, 4):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[1],
                ecolor=ecolor,
            )

        for i in range(4, 7):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[3],
                ecolor=ecolor,
            )

        for i in range(7, 10):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=lighten_color(pal[2], 1.1),
                ecolor=ecolor,
            )

        for i in range(10, 12):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[1],
                ecolor=ecolor,
            )

        for i in range(12, 14):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[3],
                ecolor=ecolor,
            )

        for i in range(14, 16):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=lighten_color(pal[2], 1.1),
                ecolor=ecolor,
            )

        for i in range(16, 18):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[1],
                ecolor=ecolor,
            )

        for i in range(18, 20):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[3],
                ecolor=ecolor,
            )

        ax.bar(
            "20",
            kc_mean_std[util.remapping_dict[20]][0],
            yerr=kc_mean_std[util.remapping_dict[20]][1],
            color=pal[1],
            ecolor=ecolor,
        )

        ax.bar(
            "21",
            kc_mean_std[util.remapping_dict[21]][0],
            yerr=kc_mean_std[util.remapping_dict[21]][1],
            color=pal[3],
            ecolor=ecolor,
        )

        # ax.axhline(2.5, color="k")
        # ax.axhline(5.5, color="k")
        # ax.axhline(8.5, color="k")
        # ax.axhline(14.5, color="k")
        # ax.axhline(18.5, color="k")

        ax.set_ylabel(r"$K_c$ ($k_BT$)")
        ax.set_xlabel(r"System")

        ax.set_ylim(0, 25)         
        ax.set_yticks(np.arange(0,30,5))

        # ax.set_xticklabels(
        #     ax.get_xticks(),
        # )

        # x_ticks_labels = [f"{sim}" for sim in range(1,22)]

        # Set number of ticks for x-axis
        # ax.set_xticks(range(21))
        # Set ticks labels for x-axis
        # ax.set_xticklabels(x_ticks_labels)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)

        fig.tight_layout()

        save_fig(fig, curr_fig_path / f"estimated_kcs_vertical{style_ext}")

        if show_figs:
            plt.show()

        fig.clear()
        plt.close(fig)


# %%
show_figs = True
curr_fig_path = Path("Figures")
curr_fig_path.mkdir(parents=True, exist_ok=True)


for style, style_ext in plot_styles:
    with plt.style.context(style):
        if style_ext:
            ecolor='white'
        else:
            ecolor='black'
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

        for i in range(1, 4):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[1],
                ecolor=ecolor,
            )

        for i in range(4, 7):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[3],
                ecolor=ecolor,
            )

        for i in range(7, 10):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=lighten_color(pal[2], 1.1),
                ecolor=ecolor,
            )

        # for i in range(10, 12):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=pal[1],
        #     )

        # for i in range(12, 14):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=pal[3],
        #     )

        # for i in range(14, 16):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=lighten_color(pal[2], 1.1),
        #     )

        # for i in range(16, 18):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=pal[1],
        #     )

        # for i in range(18, 20):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=pal[3],
        #     )

        # ax.bar(
        #     "20",
        #     kc_mean_std[util.remapping_dict[20]][0],
        #     yerr=kc_mean_std[util.remapping_dict[20]][1],
        #     color=pal[1],
        # )

        # ax.bar(
        #     "21",
        #     kc_mean_std[util.remapping_dict[21]][0],
        #     yerr=kc_mean_std[util.remapping_dict[21]][1],
        #     color=pal[3],
        # )

        # ax.axhline(2.5, color="k")
        # ax.axhline(5.5, color="k")
        # ax.axhline(8.5, color="k")
        # ax.axhline(14.5, color="k")
        # ax.axhline(18.5, color="k")

        ax.set_ylabel(r"$K_c$ ($k_BT$)")
        ax.set_xlabel(r"System")

        ax.set_ylim(0, 25)
        ax.set_yticks(np.arange(0,30,5))
        

        # ax.set_xticklabels(
        #     ax.get_xticks(),
        # )

        # x_ticks_labels = [f"{sim}" for sim in range(1,22)]

        # Set number of ticks for x-axis
        # ax.set_xticks(range(21))
        # Set ticks labels for x-axis
        # ax.set_xticklabels(x_ticks_labels)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)

        fig.tight_layout()

        save_fig(fig, curr_fig_path / f"estimated_kcs1-9{style_ext}")

        if show_figs:
            plt.show()

        fig.clear()
        plt.close(fig)


# %%
show_figs = True
curr_fig_path = Path("Figures")
curr_fig_path.mkdir(parents=True, exist_ok=True)


for style, style_ext in plot_styles:
    with plt.style.context(style):
        if style_ext:
            ecolor='white'
        else:
            ecolor='black'
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        # for i in range(1, 4):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=pal[1],
        #     )

        # for i in range(4, 7):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=pal[3],
        #     )

        # for i in range(7, 10):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=lighten_color(pal[2], 1.1),
        #     )

        for i in range(10, 12):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[1],
                ecolor=ecolor,
            )

        for i in range(12, 14):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[3],
                ecolor=ecolor,
            )

        for i in range(14, 16):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=lighten_color(pal[2], 1.1),
                ecolor=ecolor,
            )

        # for i in range(16, 18):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=pal[1],
        #     )

        # for i in range(18, 20):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=pal[3],
        #     )

        # ax.bar(
        #     "20",
        #     kc_mean_std[util.remapping_dict[20]][0],
        #     yerr=kc_mean_std[util.remapping_dict[20]][1],
        #     color=pal[1],
        # )

        # ax.bar(
        #     "21",
        #     kc_mean_std[util.remapping_dict[21]][0],
        #     yerr=kc_mean_std[util.remapping_dict[21]][1],
        #     color=pal[3],
        # )

        # ax.axhline(2.5, color="k")
        # ax.axhline(5.5, color="k")
        # ax.axhline(8.5, color="k")
        # ax.axhline(14.5, color="k")
        # ax.axhline(18.5, color="k")

        ax.set_ylabel(r"$K_c$ ($k_BT$)")
        ax.set_xlabel(r"System")

        ax.set_ylim(0, 25) 
        ax.set_yticks(np.arange(0,30,5))

        # ax.set_xticklabels(
        #     ax.get_xticks(),
        # )

        # x_ticks_labels = [f"{sim}" for sim in range(1,22)]

        # Set number of ticks for x-axis
        # ax.set_xticks(range(21))
        # Set ticks labels for x-axis
        # ax.set_xticklabels(x_ticks_labels)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)

        fig.tight_layout()

        save_fig(fig, curr_fig_path / f"estimated_kcs10-15{style_ext}")

        if show_figs:
            plt.show()

        fig.clear()
        plt.close(fig)


# %%
show_figs = True
curr_fig_path = Path("Figures")
curr_fig_path.mkdir(parents=True, exist_ok=True)


for style, style_ext in plot_styles:
    with plt.style.context(style):
        if style_ext:
            ecolor='white'
        else:
            ecolor='black'

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

        # for i in range(1, 4):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=pal[1],
        #     )

        # for i in range(4, 7):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=pal[3],
        #     )

        # for i in range(7, 10):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=lighten_color(pal[2], 1.1),
        #     )

        # for i in range(10, 12):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=pal[1],
        #     )

        # for i in range(12, 14):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=pal[3],
        #     )

        # for i in range(14, 16):
        #     ax.bar(
        #         str(i),
        #         kc_mean_std[util.remapping_dict[i]][0],
        #         yerr=kc_mean_std[util.remapping_dict[i]][1],
        #         color=lighten_color(pal[2], 1.1),
        #     )

        for i in range(16, 18):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[1],
                ecolor=ecolor,
            )

        for i in range(18, 20):
            ax.bar(
                str(i),
                kc_mean_std[util.remapping_dict[i]][0],
                yerr=kc_mean_std[util.remapping_dict[i]][1],
                color=pal[3],
                ecolor=ecolor,
            )

        ax.bar(
            "20",
            kc_mean_std[util.remapping_dict[20]][0],
            yerr=kc_mean_std[util.remapping_dict[20]][1],
            color=pal[1],
            ecolor=ecolor,
        )

        ax.bar(
            "21",
            kc_mean_std[util.remapping_dict[21]][0],
            yerr=kc_mean_std[util.remapping_dict[21]][1],
            color=pal[3],
            ecolor=ecolor,
        )

        # ax.axhline(2.5, color="k")
        # ax.axhline(5.5, color="k")
        # ax.axhline(8.5, color="k")
        # ax.axhline(14.5, color="k")
        # ax.axhline(18.5, color="k")

        ax.set_ylabel(r"$K_c$ ($k_BT$)")
        ax.set_xlabel(r"System")

        ax.set_ylim(0, 25) 
        ax.set_yticks(np.arange(0,30,5))

        # ax.set_xticklabels(
        #     ax.get_xticks(),
        # )

        # x_ticks_labels = [f"{sim}" for sim in range(1,22)]

        # Set number of ticks for x-axis
        # ax.set_xticks(range(21))
        # Set ticks labels for x-axis
        # ax.set_xticklabels(x_ticks_labels)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)

        fig.tight_layout()

        save_fig(fig, curr_fig_path / f"estimated_kcs16-21{style_ext}")

        if show_figs:
            plt.show()

        fig.clear()
        plt.close(fig)


# %%
areas = {}

for sim in util.simulations:
    gro = util.analysis_path / f"{sim}/po4_only.gro"
    traj = util.analysis_path / f"{sim}/po4_all.xtc"

    u = MDAnalysis.Universe(gro, str(traj), refresh_offsets=True)
    dims = [u.dimensions[0] for ts in u.trajectory]
    print(
        f"{sim}: mean {np.mean(dims)}, min {np.min(dims)}, max {np.max(dims)} Angstroms"
    )
    areas[sim] = np.mean(dims) ** 2


# %%
