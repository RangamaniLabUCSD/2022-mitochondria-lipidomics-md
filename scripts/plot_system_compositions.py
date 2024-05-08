# %%


# %%
import pickle
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import numpy.typing as npt
import pandas as pd

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
for sim in util.simulations:
    print(sim, util.system_compositions[int(sim)])


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

# light_factor = 0.8

# p = sns.color_palette('colorblind')

# palette = [lighten_color(p[i], j) for i, j in [(7, 1),(7, light_factor),(8,1),(8,light_factor),(0,1),(0,light_factor),(2,1),(2,light_factor)]]

# sns.palplot(palette)
# sns.palplot(p)


# %%
show_figs = True
curr_fig_path = Path("Figures/")
curr_fig_path.mkdir(parents=True, exist_ok=True)

for style, style_ext in plot_styles:
    with plt.style.context(style):
        if style_ext:
            ecolor = "white"
        else:
            ecolor = "black"

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        lipid_names = ["POPC", "DOPC", "POPE", "DOPE", "CDL1", "CDL2", "POPG", "DOPG"]

        pal = sns.color_palette("colorblind")
        light_factor = 0.6
        bar_props = [
            (7, 1, None),
            (7, light_factor, "///"),
            (8, 1, None),
            (8, light_factor, "///"),
            (1, 1, None),
            (3, 1, None),
            (2, 1.1, None),
            (2, 0.95, "///"),
        ]


        for sim in util.simulations:
            if int(sim) not in util.remapping_dict:
                continue

            composition = util.system_compositions[util.remapping_dict[int(sim)]]
            p = list()
            values = [composition[lipid] for lipid in lipid_names]
            for i, value in enumerate(values):
                b = bar_props[i]
                p.append(
                    ax.barh(
                        sim,
                        value,
                        left=np.sum(values[0:i]),
                        color=lighten_color(pal[b[0]], b[1]),
                        hatch=b[2],
                        linewidth=1,
                        edgecolor='k',
                        height=0.8,
                    )
                )

            # p = plt.barh(sim, values)
            # for lipid in lipid_names[1:-1]:
            #     p = plt.barh(sim, composition[lipid], left=p)

        # ax.axhline(2.5, color="k")
        # ax.axhline(5.5, color="k")
        # ax.axhline(8.5, color="k")
        # ax.axhline(14.5, color="k")
        # ax.axhline(18.5, color="k")

        # ax.set_ylim(1,21)
        ax.set_yticks(range(1,22))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_ylabel("System")
        ax.set_xlabel("% Composition")

        fig.tight_layout()


        # # Shrink current axis by 20%
        box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width, box.height*0.9])
        ax.legend(p, lipid_names, loc="upper center", ncols=8, bbox_to_anchor=(0.5,1.05))

        save_fig(fig, curr_fig_path/f"remapped_Compositions{style_ext}")

        if show_figs:
            plt.show()

        fig.clear()
        plt.close(fig)


# %%
show_figs = True
curr_fig_path = Path("Figures/")
curr_fig_path.mkdir(parents=True, exist_ok=True)



for style, style_ext in plot_styles:
    with plt.style.context(style):
        if style_ext:
            ecolor = "white"
        else:
            ecolor = "black"

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        lipid_names = ["POPC", "DOPC", "POPE", "DOPE", "CDL1", "CDL2", "POPG", "DOPG"]

        pal = sns.color_palette("colorblind")
        light_factor = 0.6
        bar_props = [
            (7, 1, None),
            (7, light_factor, "///"),
            (8, 1, None),
            (8, light_factor, "///"),
            (1, 1, None),
            (3, 1, None),
            (2, 1.1, None),
            (2, 0.95, "///"),
        ]


        for sim in util.simulations:
            if int(sim) not in util.remapping_dict:
                continue

            composition = util.system_compositions[util.remapping_dict[int(sim)]]
            p = list()
            values = [composition[lipid] for lipid in lipid_names]
            for i, value in enumerate(values):
                b = bar_props[i]
                p.append(
                    ax.bar(
                        sim,
                        value,
                        bottom=np.sum(values[0:i]),
                        color=lighten_color(pal[b[0]], b[1]),
                        hatch=b[2],
                        linewidth=1,
                        edgecolor='k',
                        # height=0.8,
                    )
                )

            # p = plt.barh(sim, values)
            # for lipid in lipid_names[1:-1]:
            #     p = plt.barh(sim, composition[lipid], left=p)

        # ax.axhline(2.5, color="k")
        # ax.axhline(5.5, color="k")
        # ax.axhline(8.5, color="k")
        # ax.axhline(14.5, color="k")
        # ax.axhline(18.5, color="k")

        ax.set_xticks(range(1,22))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)

        ax.set_xlabel("System")
        ax.set_ylabel("% Composition")

        fig.tight_layout()


        # # Shrink current axis by 20%
        box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width, box.height*0.9])
        ax.legend(p, lipid_names, loc="upper center", ncols=8, bbox_to_anchor=(0.5,1.08))

        save_fig(fig, curr_fig_path/f"remapped_Compositions_vert{style_ext}")

        if show_figs:
            plt.show()

        fig.clear()
        plt.close(fig)


# %%
