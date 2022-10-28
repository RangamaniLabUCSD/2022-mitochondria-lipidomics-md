import pickle
import warnings

import MDAnalysis
import numpy as np
from membrane_curvature.base import MembraneCurvature
from tqdm.contrib.concurrent import process_map

import util

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

step = 0.5  # nm


def _calculate_spectrum(args):
    gro, traj, selection, path = args
    u = MDAnalysis.Universe(gro, str(traj), continuous=True)

    dims = [u.dimensions[0] for ts in u.trajectory]
    min_dimension = np.min(dims)
    # mean_dimension = np.mean(dims)
    # max_dimension = np.max(dims)

    n_bins = int(min_dimension / 10.0 / step)  # nanometer bin spacing
    mc = MembraneCurvature(
        u,
        select=selection,
        n_x_bins=n_bins,
        n_y_bins=n_bins,
        x_range=(0, n_bins * step * 10),
        y_range=(0, n_bins * step * 10),
        wrap=False,
        interpolate=True,
    ).run(verbose=True)

    with open(path / "membrane_curvature.pickle", "wb") as handle:
        pickle.dump(mc, handle, protocol=pickle.HIGHEST_PROTOCOL)


jobs = []

# Iterate over simulations to process
for sim in util.simulations:
    # print(f"Processing: {sim}")
    gro = util.analysis_path / sim / "po4_only.gro"
    traj = util.analysis_path / f"{sim}/po4_all.xtc"
    # traj = [util.analysis_path / sim / f"po4_{i}.xtc" for i in range(1, 6)]

    t = (gro, traj, "name PO4 GL0", util.analysis_path / sim)
    # print(t)
    jobs.append(t)

r = process_map(_calculate_spectrum, jobs, max_workers=7)
