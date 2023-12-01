# import MDAnalysis as mda
import pickle
import tempfile
from functools import partial
from pathlib import Path

# import moviepy.editor as mpy
import numpy as np

# import pandas as pd
import seaborn as sns

# from scipy import integrate, interpolate
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


import matplotlib.animation as animation
import matplotlib.pyplot as plt

from plot_helper import *
import util

import pdb

# mpl.use("TkAgg")

color_palette = sns.color_palette("colorblind")

plot_styles = [
    ("./white_background_ctl.mplstyle", ""),
    ("./dark_background_ctl.mplstyle", "_dark"),
]


# mc = {}
# for sim in util.simulations:
#     with open(util.analysis_path / sim / "mc_noobject.pickle", "rb") as handle:
#         mc[sim] = pickle.load(handle)

# with open(util.analysis_path / "mc_noobject.pickle", "rb") as handle:
with open("mc_noobject_2nm.pickle", "rb") as handle:
    mc = pickle.load(handle)

# other_measures = [
#     ("thickness", "Thickness (nm)", (2.5, 6)),
#     ("height", "Height (nm)", (-3, 3)),
# ]
other_measures = []

leaflets = [("upper", "Upper"), ("lower", "Lower")]

cm = "viridis"

## KEYS IN MC
# dict_keys(['P', 'Q', 'frames', 'interpolate', 'n_frames', 'n_x_bins', 'n_y_bins', 'qx', 'qy', 'z_surface', 'mean', 'gaussian', 'thickness', 'height', 'height_power_spectrum', 'start', 'step', 'stop', 'times', 'wrap', 'x_range', 'x_step', 'y_range', 'y_step'])

_s = 10  # Average block window

frame_dt = 1  # picoseconds

# pdb.set_trace()


def make_movie(sim):
    print("Processing system:", sim)
    frame_dt = mc[sim]["times"][1] - mc[sim]["times"][0]  # picoseconds

    _b = 0
    _e = mc[sim]["n_frames"]

    n_frames = len(range(_b, _e + 1, _s))
    times = np.arange(0, _e, _s, dtype=int) * frame_dt
    shape = mc[sim]["P"].shape
    # dx = mc[sys].x_step  # Angstroms
    # dy = mc[sys].y_step  # Angstroms

    # # # ## PLOT OTHER MEASURES
    # for measure_type, measure_label, measure_range in other_measures:
    #     print("\t analyzing:", measure_type)
    #     i = 0

    #     split_indices = np.arange(_s, _e, _s, dtype=int)

    #     data = np.fromiter(
    #         map(
    #             partial(np.mean, axis=0),
    #             np.split(mc[sim][measure_type], split_indices),
    #         ),
    #         dtype=np.dtype((np.double, (shape))),
    #     )
    #     # print(mc[sim].mean())
    #     print(data.max(), data.min(), data.mean())

    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))

    #     im = ax.imshow(
    #         data[i] / 10,
    #         interpolation="gaussian",
    #         cmap=cm,
    #         origin="lower",
    #         vmin=measure_range[0],
    #         vmax=measure_range[1],
    #     )

    #     ax.set_aspect("equal")
    #     title_text = ax.set_title(f"{util.system_names[sim]}")
    #     time_text = ax.text(0.75, 0.9, "", transform=ax.transAxes)

    #     ax.axis("off")
    #     cbar = plt.colorbar(
    #         im,
    #         # ticks=tcs,
    #         orientation="horizontal",
    #         ax=ax,
    #         shrink=0.7,
    #         aspect=10,
    #         pad=0.05,
    #     )
    #     cbar.ax.tick_params(labelsize=4, width=0.5)
    #     cbar.set_label(measure_label, fontsize=6, labelpad=2)

    #     plt.tight_layout()

    #     def animate(i):
    #         im.set_data(data[i] / 10)
    #         time_text.set_text(f"{times[i]*1e-6:0.3f} μs")
    #         return (
    #             im,
    #             time_text,
    #         )

    #     interactive = False

    #     if interactive:
    #         ## INTERACTIVE ANIMATION
    #         ani = animation.FuncAnimation(
    #             fig, animate, np.arange(0, n_frames, 1), interval=50, blit=True
    #         )
    #         plt.show()
    #     else:
    #         base_dir = Path(f"render_movies/{util.sim_to_final_index[int(sim)]}/{measure_type}")
    #         base_dir.mkdir(parents=True, exist_ok=True)
    #         for i in tqdm(range(0, n_frames, 1), desc="Rendering Images"):
    #             animate(i)
    #             fig.savefig(base_dir / f"frame_{measure_type}_{i:05d}.png")

    #     ax.clear()
    #     plt.close(fig)

    ## PLOT CURVATURES
    n_frames = len(range(_b, _e + 1, _s))
    shape = mc[sim]["P"].shape

    h = mc[sim]["height"]  # Height in A
    mean = np.zeros_like(h)  # Overall mean curvature of bilayer
    lower_mean = np.zeros_like(h)
    upper_mean = np.zeros_like(h)

    for i in range(h.shape[0]):
        mean[i] = util.mean_curvature(h[i], mc[sim]["x_step"])
        lower_mean[i] = util.mean_curvature(
            mc[sim]["z_surface"]["lower"][i], mc[sim]["x_step"]
        )
        upper_mean[i] = util.mean_curvature(
            mc[sim]["z_surface"]["upper"][i], mc[sim]["x_step"]
        )

    split_indices = np.arange(_s, _e, _s, dtype=int)

    # Overall binned mean curvature in nm
    bin_mean_data = (
        np.fromiter(
            map(
                partial(np.mean, axis=0),
                np.split(mean, split_indices),
            ),
            dtype=np.dtype((np.double, (shape))),
        )
        * 10
    )
    # Binned mean curvature of lower leaflet nm
    bin_lower_mean_data = (
        np.fromiter(
            map(
                partial(np.mean, axis=0),
                np.split(lower_mean, split_indices),
            ),
            dtype=np.dtype((np.double, (shape))),
        )
        * 10
    )
    # Binned mean curvature of upper leaflet nm
    bin_upper_mean_data = (
        np.fromiter(
            map(
                partial(np.mean, axis=0),
                np.split(upper_mean, split_indices),
            ),
            dtype=np.dtype((np.double, (shape))),
        )
        * 10
    )
    # print(
    #     util.sim_to_final_index[sim], np.std(bin_mean_data.ravel()), np.std(bin_lower_mean_data.ravel()), np.std(bin_upper_mean_data.ravel())
    # )

    vmin = -0.1
    vmax = 0.1

    for style, style_ext in plot_styles:
        with plt.style.context(style):
            if style_ext:
                ecolor = "white"
            else:
                ecolor = "black"

            fig, (ax1, ax2, ax3) = plt.subplots(
                ncols=3,
                nrows=1,
                sharey=True,
                layout="constrained",
                figsize=(8, 3),
            )

            im1 = ax1.imshow(
                bin_mean_data[0],
                extent=[0, 40, 0, 40],
                interpolation="gaussian",
                cmap="PRGn",
                # origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            # plt.colorbar(im1, ax=ax1)
            ax1.set_aspect("equal")
            ax1.set_ylabel("Y (nm)")
            ax1.set_xlabel("X (nm)")
            ax1.set_title(f"Bilayer")

            im2 = ax2.imshow(
                bin_upper_mean_data[0],
                extent=[0, 40, 0, 40],
                interpolation="gaussian",
                cmap="PRGn",
                # origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            ax2.set_aspect("equal")
            # ax2.set_ylabel("Y (nm)")
            ax2.set_xlabel("X (nm)")
            ax2.set_title(f"Upper leaflet")

            im3 = ax3.imshow(
                bin_lower_mean_data[0],
                extent=[0, 40, 0, 40],
                interpolation="gaussian",
                cmap="PRGn",
                # origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            plt.colorbar(im3, ax=ax3)
            ax3.set_aspect("equal")
            # ax3.set_ylabel("Y (nm)")
            ax3.set_xlabel("X (nm)")
            ax3.set_title(f"Lower leaflet")

            title = fig.suptitle(
                f"Sim {util.sim_to_final_index[int(sim)]} "
                + r"mean curvature nm$^{-1}$ (0 μs)"
            )

            # time_text = ax3.text(0.75, 0.9, "", transform=ax3.transAxes)
            # plt.tight_layout()

            def animate(i):
                im1.set_data(bin_mean_data[i])
                im2.set_data(bin_upper_mean_data[i])
                im3.set_data(bin_lower_mean_data[i])
                title.set_text(
                    f"Sim {util.sim_to_final_index[int(sim)]}"
                    + r" mean curvature nm$^{-1}$"
                    + f" ({times[i]*1e-6:0.3f} μs)"
                )
                return (
                    im1,
                    im2,
                    im3,
                    title,
                )

            interactive = False

            if interactive:
                ## INTERACTIVE ANIMATION
                ani = animation.FuncAnimation(
                    fig, animate, np.arange(0, n_frames, 1), interval=50, blit=True
                )
                plt.show()
            else:
                base_dir = Path(
                    f"render_movies/{util.sim_to_final_index[int(sim)]}/mean"
                )
                base_dir.mkdir(parents=True, exist_ok=True)
                for i in tqdm(range(0, n_frames, 1), desc="Rendering Images"):
                    animate(i)
                    fig.savefig(base_dir / f"frame_mean{style_ext}_{i:05d}.png")
            ax1.clear()
            ax2.clear()
            ax2.clear()
            plt.close(fig)


if __name__ == "__main__":
    ## BATCH RENDER
    r = process_map(make_movie, util.simulations, max_workers=8)
    # make_movie("1")
