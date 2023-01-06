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

mpl.use("TkAgg")

color_palette = sns.color_palette("colorblind")

# mc = {}
# for sim in util.simulations:
#     with open(util.analysis_path / sim / "mc_noobject.pickle", "rb") as handle:
#         mc[sim] = pickle.load(handle)

with open(util.analysis_path / "mc_noobject.pickle", "rb") as handle:
    mc = pickle.load(handle)

curvatures = [
    ("mean", "H (nm$^{-1}$)", (-2.5, 2.5)),
    ("gaussian", "K (nm$^{-2}$)", (-2.5, 2.5)),
]

other_measures = [
    ("thickness", "Thickness (nm)", (2.5, 6)),
    ("height", "Height (nm)", (-3, 3)),
]

leaflets = [("upper", "Upper"), ("lower", "Lower")]

cm = "viridis"

## KEYS IN MC
# dict_keys(['P', 'Q', 'frames', 'interpolate', 'n_frames', 'n_x_bins', 'n_y_bins', 'qx', 'qy', 'z_surface', 'mean', 'gaussian', 'thickness', 'height', 'height_power_spectrum', 'start', 'step', 'stop', 'times', 'wrap', 'x_range', 'x_step', 'y_range', 'y_step'])

_s = 10  # Average block window

frame_dt = 1  # picoseconds


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

    ## PLOT OTHER MEASURES
    for measure_type, measure_label, measure_range in other_measures:
        print("\t analyzing:", measure_type)
        i = 0

        split_indices = np.arange(_s, _e, _s, dtype=int)

        data = np.fromiter(
            map(
                partial(np.mean, axis=0),
                np.split(mc[sim][measure_type], split_indices),
            ),
            dtype=np.dtype((np.double, (shape))),
        )
        print(mc[sim].mean())
        print(data.max(), data.min(), data.mean())

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))

        im = ax.imshow(
            data[i] / 10,
            interpolation="gaussian",
            cmap=cm,
            origin="lower",
            vmin=measure_range[0],
            vmax=measure_range[1],
        )

        ax.set_aspect("equal")
        title_text = ax.set_title(f"{util.system_names[sim]}")
        time_text = ax.text(0.75, 0.9, "", transform=ax.transAxes)

        ax.axis("off")
        cbar = plt.colorbar(
            im,
            # ticks=tcs,
            orientation="horizontal",
            ax=ax,
            shrink=0.7,
            aspect=10,
            pad=0.05,
        )
        cbar.ax.tick_params(labelsize=4, width=0.5)
        cbar.set_label(measure_label, fontsize=6, labelpad=2)

        plt.tight_layout()

        def animate(i):
            im.set_data(data[i] / 10)
            time_text.set_text(f"{times[i]*1e-6:0.3f} μs")
            return (
                im,
                time_text,
            )

        interactive = False

        if interactive:
            ## INTERACTIVE ANIMATION
            ani = animation.FuncAnimation(
                fig, animate, np.arange(0, n_frames, 1), interval=50, blit=True
            )
            plt.show()
        else:
            base_dir = Path(f"{sim}/{measure_type}")
            base_dir.mkdir(parents=True, exist_ok=True)
            for i in tqdm(range(0, n_frames, 1), desc="Rendering Images"):
                animate(i)
                fig.savefig(base_dir / f"frame_{measure_type}_{i:05d}.png")

        ax.clear()
        plt.close(fig)

    # ## PLOT CURVATURES
    # for curvature_type, curvature_label, curvature_range in curvatures:
    #     print("\t analyzing:", curvature_type)
    #     bin_mean_data = {}

    #     n_frames = len(range(_b, _e + 1, _s))
    #     shape = mc[sim].P.shape

    #     for leaflet, _ in leaflets:
    #         print("\t leaflet:", leaflet)
    #         split_indices = np.arange(_s, _e, _s, dtype=int)
    #         # tmp = np.split(mc[i].results[curvature_type][leaflet], split_indices)

    #         bin_mean_data[leaflet] = np.fromiter(
    #             map(
    #                 partial(np.mean, axis=0),
    #                 np.split(mc[sim].results[curvature_type][leaflet], split_indices),
    #             ),
    #             dtype=np.dtype((np.double, (shape))),
    #         )
    #         # print(bin_mean_data[leaflet].shape)

    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         print(tmpdirname)
    #         # _min = min(np.min(bin_mean_data["lower"]), np.min(bin_mean_data["upper"]))
    #         # _max = max(np.max(bin_mean_data["lower"]), np.max(bin_mean_data["upper"]))

    #         for i in tqdm(range(n_frames), desc="Rendering images"):
    #             fig, [ax1, ax2] = plt.subplots(
    #                 nrows=1, ncols=2, figsize=(5, 3), facecolor="white"
    #             )

    #             for ax, (leaflet, leaflet_name) in zip((ax1, ax2), leaflets):
    #                 im = ax.imshow(
    #                     bin_mean_data[leaflet][i],
    #                     # interpolation="gaussian",
    #                     cmap=cm,
    #                     origin="lower",
    #                     vmin=curvature_range[0],
    #                     vmax=curvature_range[1],
    #                 )
    #                 tcs = [curvature_range[0], 0, curvature_range[1]]

    #                 ax.set_aspect("equal")
    #                 ax.set_title("{} Leaflet".format(leaflet_name))
    #                 ax.axis("off")
    #                 cbar = plt.colorbar(
    #                     im,
    #                     ticks=tcs,
    #                     orientation="horizontal",
    #                     ax=ax,
    #                     shrink=0.7,
    #                     aspect=10,
    #                     pad=0.05,
    #                 )
    #                 cbar.ax.tick_params(labelsize=4, width=0.5)
    #                 cbar.set_label(curvature_label, fontsize=6, labelpad=2)

    #                 fig.suptitle(
    #                     f"{util.system_names[sim]} {bin_mean_data['times'][i]*1e-6:0.3f} μs"
    #                 )

    #             plt.tight_layout()
    #             plt.savefig(
    #                 Path(tmpdirname) / f"{sim}-{curvature_type}-{i}.png", format="png"
    #             )


if __name__ == "__main__":
    ## BATCH RENDER
    r = process_map(make_movie, util.simulations, max_workers=1)
    # make_movie("1")
