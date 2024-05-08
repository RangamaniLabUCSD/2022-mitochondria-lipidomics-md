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
    # ("./dark_background_ctl.mplstyle", "_dark"),
]

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


def make_movie(sim):
    print("Processing system:", sim)
    frame_dt = mc[sim]["times"][1] - mc[sim]["times"][0]  # picoseconds

    _b = 0
    _e = mc[sim]["n_frames"]

    n_frames = len(range(_b, _e + 1, _s))
    times = np.arange(0, _e, _s, dtype=int) * frame_dt
    shape = mc[sim]["P"].shape

    h = mc[sim]["height"]  # Height in A
    split_indices = np.arange(_s, _e, _s, dtype=int)

    # Overall binned mean curvature in nm
    bin_h_data = (
        np.fromiter(
            map(
                partial(np.mean, axis=0),
                np.split(h, split_indices),
            ),
            dtype=np.dtype((np.double, (shape))),
        )
        / 10
    )

    print(np.min(bin_h_data), np.max(bin_h_data), np.mean(bin_h_data), np.std(bin_h_data))

    vmin = -2  
    vmax = 2

    for style, style_ext in plot_styles:
        with plt.style.context(style):
            if style_ext:
                ecolor = "white"
            else:
                ecolor = "black"

            fig, (ax1) = plt.subplots(
                ncols=1,
                nrows=1,
                sharey=True,
                layout="constrained",
                figsize=(3, 3),
            )

            im1 = ax1.imshow(
                bin_h_data[0],
                extent=[0, 40, 0, 40],
                interpolation="gaussian",
                cmap="PRGn",
                # origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            cbar = plt.colorbar(im1, ax=ax1)
            cbar.ax.get_yaxis().labelpad = 10
            cbar.ax.set_ylabel("height (nm)", rotation=270)
            ax1.set_aspect("equal")
            ax1.set_ylabel("Y (nm)")
            ax1.set_xlabel("X (nm)")
            title = ax1.set_title(
                f"Sim {util.sim_to_final_index[int(sim)]} " + r"(0 μs)"
            )

            # title = fig.suptitle(
            #     f"Sim {util.sim_to_final_index[int(sim)]} " + r"height nm (0 μs)"
            # )

            # time_text = ax3.text(0.75, 0.9, "", transform=ax3.transAxes)
            # plt.tight_layout()

            def animate(i):
                im1.set_data(bin_h_data[i])
                title.set_text(
                    f"Sim {util.sim_to_final_index[int(sim)]} "
                    + f"({times[i]*1e-6:0.3f} μs)"
                )
                return (
                    im1,
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
                    f"render_movies/{util.sim_to_final_index[int(sim)]}/height"
                )
                base_dir.mkdir(parents=True, exist_ok=True)
                for i in tqdm(range(0, n_frames, 1), desc="Rendering Images"):
                    animate(i)
                    fig.savefig(base_dir / f"frame_height{style_ext}_{i:05d}.png")
                    fig.savefig(base_dir / f"frame_height{style_ext}_{i:05d}.pdf")
            ax1.clear()
            plt.close(fig)


if __name__ == "__main__":
    ## BATCH RENDER
    # r = process_map(make_movie, util.simulations, max_workers=8)
    make_movie(12)
