#!/usr/bin/env python
import pickle
import numpy as np
from functools import partial
import MDAnalysis

from pathlib import Path

import numpy.typing as npt

from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

import util
from plot_helper import *

from MDAnalysis.analysis.leaflet import LeafletFinder


# Location to save the final data
curvature_correlation_fd = util.analysis_path / "curvature_correlation.pickle"


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

def compute_correlation(sim):
    all_data = {}
    with open(
        util.analysis_path / f"{sim}/membrane_curvature_2nm.pickle", "rb"
    ) as handle:
        mc = pickle.load(handle)

    h = mc.results["height"][1:]
    mean = np.zeros_like(h)
    for i in range(h.shape[0]):
        mean[i] = util.mean_curvature(h[i], mc.x_step) * 10

    gro = util.analysis_path / f"{sim}/po4_only.gro"
    traj = util.analysis_path / f"{sim}/po4_all.xtc"

    u = MDAnalysis.Universe(gro, str(traj), refresh_offsets=True)
    ag = determine_leaflets(u, po4_neighbor_sel)

    all_upper = ag["upper"]
    all_lower = ag["lower"]
    # print(len(all_upper), len(all_lower))

    for lipid, query in queries.items():
        lipid_upper = ag["upper"].select_atoms(query)
        lipid_lower = ag["lower"].select_atoms(query)

        # print(lipid, len(lipids_upper), len(lipids_lower))
        if len(lipid_upper) == 0:
            continue

        gx = np.arange(mc.x_range[0], mc.x_range[1], mc.x_step)
        gy = np.arange(mc.y_range[0], mc.y_range[1], mc.y_step)
        gxm = get_midpoints(gx)
        gym = get_midpoints(gy)

        hs = []  # Curvatures for specific lipid
        ahs = []  # Curvatures for all lipids
        mhs = []  # Curvatures for all mesh points

        for i, ts in tqdm(enumerate(u.trajectory[1:]), total=len(u.trajectory[1:])):
            # Get x, y points of interest
            px = lipid_upper.positions[:, 0]
            py = lipid_upper.positions[:, 1]
            # get indices of closest mesh point
            pxd = np.digitize(px, gxm)
            pyd = np.digitize(py, gym)
            hs.extend(list(mean[i][pxd, pyd]))

            ax = all_upper.positions[:, 0]
            ay = all_upper.positions[:, 1]
            axd = np.digitize(ax, gxm)
            ayd = np.digitize(ay, gym)
            ahs.extend(list(mean[i][axd, ayd]))

            mhs.extend(mean[i].ravel())

            px = lipid_lower.positions[:, 0]
            py = lipid_lower.positions[:, 1]
            # get indices of closest mesh point
            pxd = np.digitize(px, gxm)
            pyd = np.digitize(py, gym)
            hs.extend(list(-mean[i][pxd, pyd]))

            ax = all_lower.positions[:, 0]
            ay = all_lower.positions[:, 1]
            axd = np.digitize(ax, gxm)
            ayd = np.digitize(ay, gym)
            ahs.extend(list(-mean[i][axd, ayd]))

            mhs.extend(-mean[i].ravel())

        hs = np.array(hs)
        ahs = np.array(ahs)
        mhs = np.array(mhs)

        all_data[lipid] = hs
    all_data["all"] = ahs
    all_data["mhs"] = mhs

    return (sim, all_data)


if __name__ == "__main__":
    r = dict(
        process_map(
            compute_correlation,
            np.concatenate((util.simulations, ["1_vbt"])),
            max_workers=24,
        )
    )
    print(f"Saving data to {curvature_correlation_fd}")
    with open(curvature_correlation_fd, "wb") as handle:
        pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # for sim in np.concatenate((util.simulations, ["1_vbt"])):
    # print(sim)
    # all_data[sim] = {}

    # with open(
    #     util.analysis_path / f"{sim}/membrane_curvature_2nm.pickle", "rb"
    # ) as handle:
    #     mc = pickle.load(handle)

    # h = mc.results["height"][1:]
    # mean = np.zeros_like(h)
    # for i in range(h.shape[0]):
    #     mean[i] = util.mean_curvature(h[i], mc.x_step) * 10

    # gro = util.analysis_path / f"{sim}/po4_only.gro"
    # traj = util.analysis_path / f"{sim}/po4_all.xtc"

    # u = MDAnalysis.Universe(gro, str(traj), refresh_offsets=True)
    # ag = determine_leaflets(u, po4_neighbor_sel)

    # all_upper = ag["upper"]
    # all_lower = ag["lower"]
    # # print(len(all_upper), len(all_lower))

    # for lipid, query in queries.items():
    #     lipid_upper = ag["upper"].select_atoms(query)
    #     lipid_lower = ag["lower"].select_atoms(query)

    #     # print(lipid, len(lipids_upper), len(lipids_lower))
    #     if len(lipid_upper) == 0:
    #         continue

    #     gx = np.arange(mc.x_range[0], mc.x_range[1], mc.x_step)
    #     gy = np.arange(mc.y_range[0], mc.y_range[1], mc.y_step)
    #     gxm = get_midpoints(gx)
    #     gym = get_midpoints(gy)

    #     hs = []  # Curvatures for specific lipid
    #     ahs = []  # Curvatures for all lipids

    #     for i, ts in tqdm(enumerate(u.trajectory[1:]), total=len(u.trajectory[1:])):
    #         # Get x, y points of interest
    #         px = lipid_upper.positions[:, 0]
    #         py = lipid_upper.positions[:, 1]
    #         # get indices of closest mesh point
    #         pxd = np.digitize(px, gxm)
    #         pyd = np.digitize(py, gym)
    #         hs.extend(list(mean[i][pxd, pyd]))

    #         ax = all_upper.positions[:, 0]
    #         ay = all_upper.positions[:, 1]
    #         axd = np.digitize(ax, gxm)
    #         ayd = np.digitize(ay, gym)
    #         ahs.extend(list(mean[i][axd, ayd]))

    #         px = lipid_lower.positions[:, 0]
    #         py = lipid_lower.positions[:, 1]
    #         # get indices of closest mesh point
    #         pxd = np.digitize(px, gxm)
    #         pyd = np.digitize(py, gym)
    #         hs.extend(list(-mean[i][pxd, pyd]))

    #         ax = all_lower.positions[:, 0]
    #         ay = all_lower.positions[:, 1]
    #         axd = np.digitize(ax, gxm)
    #         ayd = np.digitize(ay, gym)
    #         ahs.extend(list(-mean[i][axd, ayd]))

    #     hs = np.array(hs)
    #     ahs = np.array(ahs)

    #     fig, ax = plt.subplots(1, 1, figsize=(3, 3))  # sharex=True,

    #     # Specific lipid
    #     hsn, hs_bins, _ = ax.hist(
    #         hs, bins=100, range=[-10, 10], density=True, histtype="step", color="r"
    #     )
    #     all_data[sim][lipid] = (hsn, hs_bins)

    #     # ALL lipids
    #     asn, as_bins, _ = ax.hist(
    #         ahs,
    #         bins=100,
    #         range=[-10, 10],
    #         density=True,
    #         color="k",
    #         alpha=0.7,
    #     )
    #     all_data[sim]["all"] = (asn, as_bins)

    #     print(f"Overall mean: {np.mean(ahs)}; {lipid} mean {np.mean(hs)}")

    #     ax.axvline(0, color="k", linestyle="--", linewidth=1)

    #     ax.set_xlabel(r"Mean curvature (/nm)")
    #     ax.set_ylabel(r"Density")

    #     if sim == "1_vbt":
    #         ax.set_title(f"1_vbt {lipid}")
    #     else:
    #         ax.set_title(f"{util.sim_to_final_index[int(sim)]} {lipid}")
    #     ax.set_xlim(-10, 10)

    #     # ax.legend(loc="upper right")

    #     # # Shrink current axis by 20%
    #     # box = ax.get_position()
    #     # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    #     # # Put a legend to the right of the current axis
    #     # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    #     fig.tight_layout()

    #     if sim == "1_vbt":
    #         fig.savefig(curr_fig_path / f"1_vbt_{lipid}.png", format="png")
    #         fig.savefig(curr_fig_path / f"1_vbt_{lipid}.pdf", format="pdf")
    #     else:
    #         fig.savefig(
    #             curr_fig_path / f"{util.sim_to_final_index[int(sim)]}_{lipid}.png",
    #             format="png",
    #         )
    #         fig.savefig(
    #             curr_fig_path / f"{util.sim_to_final_index[int(sim)]}_{lipid}.pdf",
    #             format="pdf",
    #         )

    #     if show_figs:
    #         plt.show()

    #     fig.clear()
    #     plt.close(fig)
