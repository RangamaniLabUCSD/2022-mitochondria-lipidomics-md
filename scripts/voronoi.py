from functools import partialmethod

import matplotlib.pyplot as plt
import MDAnalysis
from MDAnalysis.analysis.leaflet import LeafletFinder
import numpy as np
from scipy.spatial import Voronoi
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

import pickle

import util

# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

lipids = ["POPC", "DOPC", "POPE", "DOPE", "CDL1", "POPG", "DOPG"]

lipid_dict = dict([[j, i] for i, j in enumerate(lipids)])

leaflets = ["upper", "lower"]


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


def run_voronoi(sim):
    gro = util.analysis_path / f"{sim}/po4_only.gro"
    traj = [util.analysis_path / f"{sim}/po4_{i}.xtc" for i in [4, 5]]

    u = MDAnalysis.Universe(gro, *map(str, traj))

    ag = determine_leaflets(u, "name PO4 or name GL0")

    count_dict = {
        "upper": np.zeros((len(u.trajectory), 7, 7)),
        "lower": np.zeros((len(u.trajectory), 7, 7)),
        "name_map": lipid_dict,
    }

    for ts in tqdm(u.trajectory, desc=f"{sim}", position=int(sim)):
        for leaflet in leaflets:
            vor = Voronoi(ag[leaflet].positions[:, 0:-1])
            v = np.array(
                [
                    [lipid_dict[u.atoms[i].resname], lipid_dict[u.atoms[j].resname]]
                    for i, j in vor.ridge_points
                ]
            )
            v.sort(axis=1)

            for i, j in v:
                count_dict[leaflet][ts.frame, i, j] += 1

    with open(util.analysis_path / f"{sim}/voronoi_leaflet_glo.pickle", "wb") as handle:
        pickle.dump(count_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_neighbor_search(sim):
    gro = util.analysis_path / f"{sim}/po4_only.gro"
    traj = [util.analysis_path / f"{sim}/po4_{i}.xtc" for i in [4, 5]]

    u = MDAnalysis.Universe(gro, *map(str, traj))

    ag = determine_leaflets(u, "name PO4 or name GL0")

    # print(len(u.trajectory))
    count_dict = {
        "upper": np.zeros((len(u.trajectory), 7, 7)),
        "lower": np.zeros((len(u.trajectory), 7, 7)),
        "name_map": lipid_dict,
    }

    for ts in tqdm(u.trajectory, desc=f"{sim}", position=int(sim)):
        for leaflet in leaflets:
            for atom in ag[leaflet].atoms:
                i = lipid_dict[atom.resname]
                curr = MDAnalysis.AtomGroup([atom])

                sel = ag[leaflet].select_atoms("around 15 group curr", curr=curr, updating=True)
                for res in sel.residues:
                    j = lipid_dict[res.resname]

                    if i <= j:
                        count_dict[leaflet][ts.frame, i, j] += 1
                    else:
                        count_dict[leaflet][ts.frame, j, i] += 1
    with open(
        util.analysis_path / f"{sim}/neighbor_enrichment_leaflet_glo.pickle", "wb"
    ) as handle:
        pickle.dump(count_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
# def main():
#     freeze_support()

#     tqdm.set_lock(RLock())
#     pool = Pool(processes=7, initargs=(tqdm.get_lock(),), initializer = tqdm.set_lock)

#     p = pool.map(r)
#     pool.close()

if __name__ == "__main__":
    process_map(run_voronoi, util.simulations, max_workers=7)
    process_map(run_neighbor_search, util.simulations, max_workers=7)
