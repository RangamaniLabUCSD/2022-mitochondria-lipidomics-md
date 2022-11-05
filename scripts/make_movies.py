# import MDAnalysis as mda
import pickle
import tempfile
from functools import partial
from pathlib import Path

import moviepy.editor as mpy
import numpy as np

# import pandas as pd
import seaborn as sns

# from scipy import integrate, interpolate
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from plot_helper import *
import util

color_palette = sns.color_palette("colorblind")

mc = {}
for sim in util.simulations:
    with open(util.analysis_path / sim / "membrane_curvature.pickle", "rb") as handle:
        mc[sim] = pickle.load(handle)


curvatures = [
    ("mean", "H (nm$^{-1}$)", (-2.5, 2.5)),
    ("gaussian", "K (nm$^{-2}$)", (-2.5, 2.5)),
]

other_measures = [
    ("thickness", "Thickness (nm)", (2.5, 6)),
    ("height", "Height (nm)", (-2.5, 2.5)),
]

leaflets = [("upper", "Upper"), ("lower", "Lower")]

cm = "PRGn"

## KEYS IN MC
# z_surface
# mean
# gaussian
# thickness
# height
# thickness_power_spectrum
# height_power_spectrum

_s = 10  # Average block window


def make_movie(sim):
    print("Processing system:", sim)
    frame_dt = mc[sim].times[1] - mc[sim].times[0]  # picoseconds

    _b = 0
    _e = len(mc[sim].frames)

    # dx = mc[sys].x_step  # Angstroms
    # dy = mc[sys].y_step  # Angstroms

    ## PLOT CURVATURES
    for curvature_type, curvature_label, curvature_range in curvatures:
        print("\t analyzing:", curvature_type)
        bin_mean_data = {}

        n_frames = len(range(_b, _e + 1, _s))
        shape = mc[sim].P.shape

        bin_mean_data["times"] = np.arange(0, _e, _s, dtype=int) * frame_dt  # picoseconds
        for leaflet, _ in leaflets:
            print("\t leaflet:", leaflet)
            split_indices = np.arange(_s, _e, _s, dtype=int)
            # tmp = np.split(mc[i].results[curvature_type][leaflet], split_indices)

            bin_mean_data[leaflet] = np.fromiter(
                map(
                    partial(np.mean, axis=0),
                    np.split(mc[sim].results[curvature_type][leaflet], split_indices),
                ),
                dtype=np.dtype((np.double, (shape))),
            )
            # print(bin_mean_data[leaflet].shape)

        with tempfile.TemporaryDirectory() as tmpdirname:
            print(tmpdirname)
            _min = min(np.min(bin_mean_data["lower"]), np.min(bin_mean_data["upper"]))
            _max = max(np.max(bin_mean_data["lower"]), np.max(bin_mean_data["upper"]))

            for i in tqdm(range(n_frames), desc="Rendering images"):
                fig, [ax1, ax2] = plt.subplots(
                    nrows=1, ncols=2, figsize=(5, 3), facecolor="white"
                )

                for ax, (leaflet, leaflet_name) in zip((ax1, ax2), leaflets):
                    im = ax.imshow(
                        bin_mean_data[leaflet][i],
                        # interpolation="gaussian",
                        cmap=cm,
                        origin="lower",
                        vmin=curvature_range[0],
                        vmax=curvature_range[1],
                    )
                    tcs = [curvature_range[0], 0, curvature_range[1]]

                    ax.set_aspect("equal")
                    ax.set_title("{} Leaflet".format(leaflet_name))
                    ax.axis("off")
                    cbar = plt.colorbar(
                        im,
                        ticks=tcs,
                        orientation="horizontal",
                        ax=ax,
                        shrink=0.7,
                        aspect=10,
                        pad=0.05,
                    )
                    cbar.ax.tick_params(labelsize=4, width=0.5)
                    cbar.set_label(curvature_label, fontsize=6, labelpad=2)

                    fig.suptitle(
                        f"{util.system_names[sim]} {bin_mean_data['times'][i]*1e-6:0.3f} μs"
                    )

                plt.tight_layout()
                plt.savefig(
                    Path(tmpdirname) / f"{sim}-{curvature_type}-{i}.png", format="png"
                )

            clip = mpy.ImageSequenceClip(
                [
                    f"{tmpdirname}/{sim}-{curvature_type}-{i}.png"
                    for i in range(n_frames)
                ],
                fps=30,
            )
            clip.write_videofile(f"Figures/{sim}_{curvature_type}.mp4", fps=30)
        del clip

    ## PLOT OTHER MEASURES
    for measure_type, measure_label, measure_range in other_measures:
        print("\t analyzing:", measure_type)
        bin_mean_data = {}

        n_frames = len(range(_b, _e + 1, _s))
        shape = mc[sim].P.shape

        bin_mean_data["times"] = np.arange(0, _e, _s, dtype=int) * frame_dt  # picoseconds

        split_indices = np.arange(_s, _e, _s, dtype=int)

        bin_mean_data["data"] = np.fromiter(
            map(
                partial(np.mean, axis=0),
                np.split(mc[sim].results[measure_type], split_indices),
            ),
            dtype=np.dtype((np.double, (shape))),
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            print(tmpdirname)
            _min = np.min(bin_mean_data["data"]) / 10
            _max = np.max(bin_mean_data["data"]) / 10

            for i in tqdm(range(n_frames), desc="Rendering images"):
                fig, ax = plt.subplots(
                    nrows=1, ncols=1, figsize=(3, 3), facecolor="white"
                )

                im = ax.imshow(
                    bin_mean_data["data"][i] / 10,
                    # interpolation="gaussian",
                    cmap=cm,
                    origin="lower",
                    vmin=measure_range[0],
                    vmax=measure_range[1],
                )

                ax.set_aspect("equal")
                ax.set_title(f"{util.system_names[sim]} {bin_mean_data['times'][i]*1e-6:0.3f} μs")
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
                plt.savefig(
                    Path(tmpdirname) / f"{sim}-{measure_type}-{i}.png", format="png"
                )

            clip = mpy.ImageSequenceClip(
                [f"{tmpdirname}/{sim}-{measure_type}-{i}.png" for i in range(n_frames)],
                fps=30,
            )
            clip.write_videofile(f"Figures/{sim}_{measure_type}.mp4", fps=30)
        del clip


r = process_map(make_movie, util.simulations, max_workers=6)
    