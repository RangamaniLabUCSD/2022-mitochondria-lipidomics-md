# ***************************************************************************
# Copyright (C) 2022
# by Christopher T. Lee
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# **************************************************************************

import importlib.util

MPL_SPEC = importlib.util.find_spec("matplotlib")
MPL_FOUND = MPL_SPEC is not None

SNS_SPEC = importlib.util.find_spec("matplotlib")
SNS_FOUND = SNS_SPEC is not None

if not MPL_FOUND:
    raise RuntimeError(f"Plotting requires Matplotlib")
if not SNS_FOUND:
    raise RuntimeError(f"Plotting requires Seaborn")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
# import matplotlib.pylab as mpl

import seaborn as sns
from operator import sub

from mpl_toolkits import axes_grid1
# from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
#                                                   mark_inset)



print("Matplotlib Version:", mpl.__version__)


plt.style.use("seaborn-colorblind")  # set plot style
mpl.rcParams["font.sans-serif"] = "Arial"
mpl.rcParams["font.family"] = "sans-serif"

DARK_GREY = (0.1, 0.1, 0.1)
THINNER_LINE = 0.5
THIN_LINE = 0.7
NORMAL_LINE = 0.8

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
FIG_DPI = 300

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["figure.figsize"] = [3, 3]  # Default 3x3
plt.rcParams["figure.dpi"] = FIG_DPI
plt.rcParams["savefig.dpi"] = FIG_DPI
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.labelpad"] = 2.5
plt.rcParams["lines.markersize"] = 1

# Force TrueType fonts
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

LinearWhiteBlueColormap = mpl.colors.LinearSegmentedColormap.from_list(
    "LinearBlues", ["white", sns.color_palette()[0]], N=256
)

# def matplotlibStyle(small: float = 6, medium: float = 8, large: float = 10):
#     """Set matplotlib plotting style

#     Args:
#         s (int, optional): Small size. Defaults to 6.
#         m (int, optional): Medium size. Defaults to 8.
#         l (int, optional): Large size. Defaults to 10.
#     """
#     plt.rcParams["font.sans-serif"] = "Arial"
#     plt.rcParams["font.family"] = "sans-serif"
#     plt.rcParams["lines.linewidth"] = 2
#     plt.rcParams["savefig.dpi"] = 600
#     # mpl.rcParams.update({'font.size': 8})
#     plt.rc("font", size=large)  # controls default text sizes
#     plt.rc("axes", titlesize=large)  # fontsize of the axes title
#     plt.rc("axes", labelsize=medium)  # fontsize of the x and y labels
#     plt.rc("xtick", labelsize=medium)  # fontsize of the tick labels
#     plt.rc("ytick", labelsize=medium)  # fontsize of the tick labels
#     plt.rc("legend", fontsize=small, frameon=False)  # legend fontsize
#     plt.rc("figure", titlesize=large)  # fontsize of the figure title
#     plt.rc("pdf", fonttype=42)


def add_colorbar(im, ax=None, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    if ax is None:
        ax = plt.gca()
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def get_aspect(ax):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio


def plot_errorbar(means, stds, c, label):

    plt.figure(figsize=(6, 4))
    #     plt.rc('font', size=20)
    #     plt.rc('xtick', labelsize=20)
    #     plt.rc('ytick', labelsize=20)

    color = c

    plt.plot(means.index, means, c=color)
    times_cur = means.index
    yminus = means - stds
    yplus = means + stds

    plt.fill_between(
        times_cur,
        np.asarray(yminus),
        np.asarray(yplus),
        alpha=0.25,
        edgecolor=color,
        facecolor=color,
        linewidth=1,
        antialiased=True,
    )

    plt.xlabel("Time (s)")
    plt.ylabel(label)

    plt.tick_params(reset=True, color="black", direction="in", length=4)
    plt.grid(False)

    plt.tight_layout()


def plot_multiple_errorbars(means, stds, c, label):
    # You can call this multiple times to make several plots on top of each other.
    #     plt.figure(figsize=(6,4))
    #     plt.rc('font', size=20)
    #     plt.rc('xtick', labelsize=20)
    #     plt.rc('ytick', labelsize=20)

    color = c

    plt.plot(means.index, means, c=color)
    times_cur = means.index
    yminus = means - stds
    yplus = means + stds

    plt.fill_between(
        times_cur,
        np.asarray(yminus),
        np.asarray(yplus),
        alpha=0.25,
        edgecolor=color,
        facecolor=color,
        linewidth=1,
        antialiased=True,
    )

    plt.xlabel("Time (s)")
    plt.ylabel(label)

    plt.tick_params(reset=True, color="black", direction="in", length=4)
    plt.grid(False)


#     plt.tight_layout()
# auto_fit the plot dimensions using plot tight layout
def plot_multiple_errorbars_tight(means, stds, c, label):

    # You can call this multiple times to make several plots on top of each other.

    #     plt.figure(figsize=(6,4))
    #     plt.rc('font', size=20)
    #     plt.rc('xtick', labelsize=20)
    #     plt.rc('ytick', labelsize=20)

    color = c

    plt.plot(means.index, means, c=color)
    times_cur = means.index
    yminus = means - stds
    yplus = means + stds

    plt.fill_between(
        times_cur,
        np.asarray(yminus),
        np.asarray(yplus),
        alpha=0.25,
        edgecolor=color,
        facecolor=color,
        linewidth=1,
        antialiased=True,
    )

    plt.xlabel("Time (s)")
    plt.ylabel(label)

    plt.tick_params(reset=True, color="black", direction="in", length=4)
    plt.grid(False)

    plt.tight_layout()

def custom_cmap(size, label, ticklabels, cmap=cm.RdPu):
    """
    Function to define colormap
    """
    cmap = cm.RdPu  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mplt.colors.LinearSegmentedColormap.from_list(
        "Custom cmap", cmaplist, cmap.N
    )
    bounds = np.linspace(0, 1, size)
    norm = mplt.colors.BoundaryNorm(bounds, cmap.N)
    val = (float(bounds[1]) - float(bounds[0])) / 2
    ticks = np.linspace(val, 1 - val, size - 1)

    cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    cb = mplt.colorbar.ColorbarBase(
        cax,
        cmap=cmap,
        norm=norm,
        spacing="proportional",
        ticks=ticks,
        boundaries=bounds,
        format="%1i",
        label=label,
    )
    cb.ax.set_yticklabels(ticklabels)  # vertically oriented colorbar
    
import datetime

now = datetime.datetime.now()
date = now.strftime("%Y%m%d")


def save_fig(fig, basename: str):
    fig.savefig(f"{basename}.png", format="png")
    fig.savefig(f"{basename}.pdf", format="pdf")
