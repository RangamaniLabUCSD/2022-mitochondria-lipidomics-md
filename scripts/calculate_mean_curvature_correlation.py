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
curvature_correlation_fd = util.analysis_path / "mean_curvature_correlation_negative.pickle"


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


def wrap_and_sanitize(pxy, ts, mc):
    """Wrap coordinates and remove values too far from closest known point

    Args:
        pxy (np.ndarray): XY coordinates
        ts: MDAnalysis timestep object
        mc: curvature object

    Returns:
        _type_: _description_
    """
    gx = mc.x_range[1] - mc.x_step
    
    # wrap PBC if point is more than half step greater than the closest data value
    if ts.dimensions[0] > mc.x_range[1] - mc.x_step / 2:
        pxy = np.where(pxy > gx + mc.x_step / 2, pxy - ts.dimensions[0], pxy)
    # Remove values which are too far from a known data point
    pxy = pxy[(pxy >= -mc.x_step / 2).all(axis=1), :]
    return pxy[:, 0], pxy[:, 1]


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
            px, py = wrap_and_sanitize(lipid_upper.positions[:, 0:2], ts, mc)
            # get indices of closest mesh point
            pxd = np.digitize(px, gxm)
            pyd = np.digitize(py, gym)
            hs.extend(list(mean[i][pxd, pyd]))

            ax, ay = wrap_and_sanitize(all_upper.positions[:, 0:2], ts, mc)
            axd = np.digitize(ax, gxm)
            ayd = np.digitize(ay, gym)
            ahs.extend(list(mean[i][axd, ayd]))

            mhs.extend(mean[i].ravel())

            px, py = wrap_and_sanitize(lipid_lower.positions[:, 0:2], ts, mc)
            # get indices of closest mesh point
            pxd = np.digitize(px, gxm)
            pyd = np.digitize(py, gym)
            hs.extend(list(-mean[i][pxd, pyd]))

            ax, ay = wrap_and_sanitize(all_lower.positions[:, 0:2], ts, mc)
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
