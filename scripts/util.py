import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder, optimize_cutoff

import re
from pathlib import Path
import numpy as np
import numpy.typing as npt

from typing import Tuple

# import jax
# import jax.numpy as jnp

import math
from scipy import stats
from functools import partial


simulations = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
sizes = ["large", "small"]

membrane_sel = "resname POPC DOPC POPE DOPE CDL1 POPG DOPG"
po4_sel = "name PO4 PO41 PO42 GL0"

system_names = {
    "1": "+CDL; 0Sat.",
    "2": "+CDL; +Sat.",
    "3": "+CDL; ++Sat.",
    "4": "-CDL; 0Sat.",
    "5": "-CDL; +Sat.",
    "6": "-CDL; ++Sat.",
    "7": "CDL Only",
    "8": "PO; CDL",
    "9": "DO; CDL",
    "10": "PO; PG",
    "11": "DO; PG",
}

lipid_names = {"POPC", "DOPC", "POPE", "DOPE", "CDL1", "POPG", "DOPG"}

archive_path = Path("/net/engram/ctlee/mito_lipidomics")
scratch_path = Path("/scratch/ctlee/mito_lipidomics_scratch")

source_control_path = Path("/home/ctlee/2022-mitochondria-lipidomics-md")

sim_path = archive_path / "sims"
scratch_sim_path = scratch_path / "sims"

mdp_path = source_control_path / "mdps"
script_path = source_control_path / "scripts"

analysis_path = scratch_path / "analysis"
analysis_archive_path = archive_path / "analysis"

analysis_large_file_path = Path("/u1/ctlee/mito_lipidomics")

gmxls_bin = Path(
    "/home/jessie.gan/gromacs-ls_install/gromacs-ls-2016.3/build/bin/gmx_LS"
)


def count_residues(u):
    count_dict = {}
    for residue in u.residues:
        if residue.resname not in count_dict:
            count_dict[residue.resname] = 1
        else:
            count_dict[residue.resname] += 1
    return count_dict


def system_report(u):
    count_dict = {}
    for residue in u.residues:
        if residue.resname not in count_dict:
            count_dict[residue.resname] = 1
        else:
            count_dict[residue.resname] += 1
    print(f"\tComposition: {count_dict}")
    assert u.atoms.total_charge() == 0
    print(f"\t     charge: {u.atoms.total_charge()}")
    print(f"\t  num atoms: {len(u.atoms)}")
    # print(f"\t dimensions: {u.dimensions[0:3]/10} nm")
    print()
    return count_dict


def _check_leaflet(u):
    # ag = u.select_atoms("resname POPC DOPC POPE DOPE")
    # u.trajectory.add_transformations(center_membrane(ag, shift=5))
    # print('Centered')
    rcutoff, n = optimize_cutoff(u, "name PO4")
    print(rcutoff, n)
    leafs = LeafletFinder(u, "name PO4", rcutoff, pbc=True)
    top = leafs.groups(0)
    bottom = leafs.groups(1)
    # leafs.write_selection('selection.vmd')

    print(len(top.residues), count_residues(top))
    print(len(bottom.residues), count_residues(bottom))

    return (set([r.ix for r in top.residues]), set([r.ix for r in bottom.residues]))


def check_leaflet(top, gro):
    u = mda.Universe(top, gro, topology_format="ITP")
    return _check_leaflet(u)


def statistical_inefficiency(
    data,
    blocks: npt.NDArray[np.int32] = np.arange(1, 257, 1),
    discards: npt.NDArray[np.int32] = np.arange(0, 100, 10),
):
    SI = np.zeros((len(discards), len(blocks)))

    for i, discard in enumerate(discards):
        # Discard front bit of data
        _, remainder = np.split(data, [int(discard / 100 * len(data))])
        _, block_var, _ = _block_average(remainder, blocks)

        SI[i] = blocks * (block_var / block_var[0])
    return discards, blocks, SI


def _block_average(
    data: npt.ArrayLike, blocks: npt.NDArray[np.int32] = np.arange(1, 100, 1)
) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]]:
    block_mean = np.zeros((len(blocks)))
    block_var = np.zeros((len(blocks)))
    block_sem = np.zeros((len(blocks)))
    for i, block in enumerate(blocks):
        split_indices = np.arange(block, len(data), block, dtype=int)
        if block > len(data):
            block_mean[i] = np.nan
            continue
        blocked_data = np.fromiter(
            map(
                partial(np.mean, axis=0),
                np.split(data, split_indices),
            ),
            dtype=np.double,
        )

        # Truncate number of blocks if not evenly divisible
        if len(data) % block:
            blocked_data = blocked_data[:-1]
        block_mean[i] = np.mean(blocked_data)
        block_var[i] = np.var(blocked_data, ddof=1)
        block_sem[i] = stats.sem(blocked_data, ddof=1)
    return block_mean, block_var, block_sem


def block_average(
    data,
    discard = 20,
    blocks: npt.NDArray[np.int32] = np.arange(1, 257, 1),
) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]]:
    _, remainder = np.split(data, [int(discard / 100 * len(data))])
    return _block_average(remainder, blocks)
