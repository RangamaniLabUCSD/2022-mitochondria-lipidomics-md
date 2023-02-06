from tokenize import Double
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder, optimize_cutoff

import re
from pathlib import Path
import numpy as np
import numpy.typing as npt

from typing import Tuple, Callable, List

import warnings

import math
from scipy import stats
from functools import partial


simulations = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
]

non_cdl_simulations = [
    "4",
    "5",
    "6",
    "10",
    "11",
]
sizes = ["large", "small"]

membrane_sel = "resname POPC DOPC POPE DOPE CDL1 CDL2 POPG DOPG"
po4_sel = "name PO4 PO41 PO42 GL0"

system_names = {
    "1": "+CL1; 0S",
    "2": "+CL1; +S",
    "3": "+CL1; ++S",
    "4": "-CL1; 0S",
    "5": "-CL1; +S",
    "6": "-CL1; ++S",
    "7": "CL1 Only",
    "8": "PO; CL1",
    "9": "DO; CL1",
    "10": "PO; PG",
    "11": "DO; PG",
    "12": "SFA3 (dcrd1) NEW",
    "13": "Itay O CL1",
    "14": "Itay I CL2",
    "15": "SFA3 CL1 NEW",
    "16": "+CL2; 0S",
    "17": "+CL2; +S",
    "18": "+CL2; ++S",
    "19": "CL2 Only",
    "20": "PO; CL2",
    "21": "DO; CL2",
    "22": "Itay O CL2",
    "23": "Itay I CL2",
    "24": "SFA3 CL2 NEW",
}

system_compositions = {
    1: {
        "POPC": 12,
        "DOPC": 46,
        "POPE": 3,
        "DOPE": 27,
        "CDL1": 12,
        "CDL2": 0,
        "POPG": 0,
        "DOPG": 0,
    },
    2: {
        "POPC": 26,
        "DOPC": 25,
        "POPE": 8,
        "DOPE": 29,
        "CDL1": 12,
        "CDL2": 0,
        "POPG": 0,
        "DOPG": 0,
    },
    3: {
        "POPC": 34,
        "DOPC": 22,
        "POPE": 14,
        "DOPE": 18,
        "CDL1": 12,
        "CDL2": 0,
        "POPG": 0,
        "DOPG": 0,
    },
    4: {
        "POPC": 10,
        "DOPC": 55,
        "POPE": 2,
        "DOPE": 22,
        "CDL1": 0,
        "CDL2": 0,
        "POPG": 6,
        "DOPG": 5,
    },
    5: {
        "POPC": 20,
        "DOPC": 34,
        "POPE": 7,
        "DOPE": 25,
        "CDL1": 0,
        "CDL2": 0,
        "POPG": 12,
        "DOPG": 2,
    },
    6: {
        "POPC": 34,
        "DOPC": 20,
        "POPE": 14,
        "DOPE": 18,
        "CDL1": 0,
        "CDL2": 0,
        "POPG": 12,
        "DOPG": 2,
    },
    7: {
        "POPC": 0,
        "DOPC": 0,
        "POPE": 0,
        "DOPE": 0,
        "CDL1": 100,
        "CDL2": 0,
        "POPG": 0,
        "DOPG": 0,
    },
    8: {
        "POPC": 50,
        "DOPC": 0,
        "POPE": 30,
        "DOPE": 0,
        "CDL1": 20,
        "CDL2": 0,
        "POPG": 0,
        "DOPG": 0,
    },
    9: {
        "POPC": 0,
        "DOPC": 50,
        "POPE": 0,
        "DOPE": 30,
        "CDL1": 20,
        "CDL2": 0,
        "POPG": 0,
        "DOPG": 0,
    },
    10: {
        "POPC": 50,
        "DOPC": 0,
        "POPE": 30,
        "DOPE": 0,
        "CDL1": 0,
        "CDL2": 0,
        "POPG": 0,
        "DOPG": 20,
    },
    11: {
        "POPC": 0,
        "DOPC": 50,
        "POPE": 0,
        "DOPE": 30,
        "CDL1": 0,
        "CDL2": 0,
        "POPG": 0,
        "DOPG": 20,
    },
    # SFA3(dcrd1) NEW
    12: {
        "POPC": 17,
        "DOPC": 26,
        "POPE": 22,
        "DOPE": 21,
        "CDL1": 0,
        "CDL2": 0,
        "POPG": 12,
        "DOPG": 2,
    },
    # ITAY OUTER LEAFLET
    13: {
        "POPC": 11,
        "DOPC": 44,
        "POPE": 3,
        "DOPE": 26,
        "CDL1": 16,
        "CDL2": 0,
        "POPG": 0,
        "DOPG": 0,
    },
    # ITAY INNER LEAFLET
    14: {
        "POPC": 12,
        "DOPC": 48,
        "POPE": 3,
        "DOPE": 28,
        "CDL1": 9,
        "CDL2": 0,
        "POPG": 0,
        "DOPG": 0,
    },
    # SFA3 (+CL) NEW
    15: {
        "POPC": 18,
        "DOPC": 26,
        "POPE": 22,
        "DOPE": 22,
        "CDL1": 12,
        "CDL2": 0,
        "POPG": 0,
        "DOPG": 0,
    },
    ### NEW CL-2 systems
    16: {
        "POPC": 12,
        "DOPC": 46,
        "POPE": 3,
        "DOPE": 27,
        "CDL1": 0,
        "CDL2": 12,
        "POPG": 0,
        "DOPG": 0,
    },
    17: {
        "POPC": 26,
        "DOPC": 25,
        "POPE": 8,
        "DOPE": 29,
        "CDL1": 0,
        "CDL2": 12,
        "POPG": 0,
        "DOPG": 0,
    },
    18: {
        "POPC": 34,
        "DOPC": 22,
        "POPE": 14,
        "DOPE": 18,
        "CDL1": 0,
        "CDL2": 12,
        "POPG": 0,
        "DOPG": 0,
    },
    19: {
        "POPC": 0,
        "DOPC": 0,
        "POPE": 0,
        "DOPE": 0,
        "CDL1": 0,
        "CDL2": 100,
        "POPG": 0,
        "DOPG": 0,
    },
    20: {
        "POPC": 50,
        "DOPC": 0,
        "POPE": 30,
        "DOPE": 0,
        "CDL1": 0,
        "CDL2": 20,
        "POPG": 0,
        "DOPG": 0,
    },
    21: {
        "POPC": 0,
        "DOPC": 50,
        "POPE": 0,
        "DOPE": 30,
        "CDL1": 0,
        "CDL2": 20,
        "POPG": 0,
        "DOPG": 0,
    },
    # ITAY OUTER LEAFLET
    22: {
        "POPC": 11,
        "DOPC": 44,
        "POPE": 3,
        "DOPE": 26,
        "CDL1": 0,
        "CDL2": 16,
        "POPG": 0,
        "DOPG": 0,
    },
    # ITAY INNER LEAFLET
    23: {
        "POPC": 12,
        "DOPC": 48,
        "POPE": 3,
        "DOPE": 28,
        "CDL1": 0,
        "CDL2": 9,
        "POPG": 0,
        "DOPG": 0,
    },
    # SFA3 (+CL) NEW
    24: {
        "POPC": 18,
        "DOPC": 26,
        "POPE": 22,
        "DOPE": 22,
        "CDL1": 0,
        "CDL2": 12,
        "POPG": 0,
        "DOPG": 0,
    },
}

production_files = [
    "production.trr",
    "production.part0002.trr",
    "production.part0003.trr",
    "production.part0004.trr",
    "production.part0005.trr",
]

lipid_names = {"POPC", "DOPC", "POPE", "DOPE", "CDL1", "CDL2" "POPG", "DOPG"}

archive_path = Path("/net/engram/ctlee/mito_lipidomics")
scratch_path = Path("/u2/ctlee/mito_lipidomics_scratch")

source_control_path = Path("/home/ctlee/2022-mitochondria-lipidomics-md")

sim_path = scratch_path / "sims"
# scratch_sim_path = scratch_path / "sims"

mdp_path = source_control_path / "mdps_continuation"
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
        if residue.resname == "ION":
            name = residue.atoms[0].name
            if name not in count_dict:
                count_dict[name] = 1
            else: 
                count_dict[name] += 1 
        else:
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
            block_var[i] = np.nan
            block_sem[i] = np.nan
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
    discard=20,
    blocks: npt.NDArray[np.int32] = np.arange(1, 257, 1),
) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]]:
    _, remainder = np.split(data, [int(discard / 100 * len(data))])
    return _block_average(remainder, blocks)


def nd_block_average(
    data: npt.ArrayLike,
    axis: int = 0,
    func: Callable[[npt.ArrayLike], float] = np.mean,
    blocks: npt.NDArray[np.int32] = np.arange(1, 100, 1),
) -> npt.ArrayLike:
    """Perform block analysis on n-dimensional data

    Args:
        data (npt.ArrayLike): _description_
        axis (int, optional): _description_. Defaults to 0.
        func (Callable[[npt.ArrayLike], float], optional): _description_. Defaults to np.mean.
        blocks (npt.NDArray[np.int32], optional): _description_. Defaults to np.arange(1, 100, 1).

    Raises:
        np.AxisError: _description_

    Returns:
        Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]]: _description_
    """

    # Guard against bad axis
    if axis >= len(data.shape):
        raise np.AxisError(axis, len(data.shape))

    result_shape = tuple([v for i, v in enumerate(data.shape) if i != axis])

    result = np.empty((len(blocks), *result_shape))
    # print("results_shape", result.shape, result_shape)

    for i, block in enumerate(blocks):
        Nb, r = divmod(data.shape[axis], block)
        split_indices = np.arange(r, data.shape[axis], block, dtype=int)

        if block > data.shape[axis]:
            result[i] = np.nan
            continue

        # Compute block average
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            blocked_data = np.fromiter(
                map(
                    partial(np.mean, axis=axis),
                    np.split(data, split_indices, axis=axis),  # List of blocks
                ),
                dtype=np.dtype((np.double, (*result_shape,))),
            )

        # Truncate first block which is either empty or has less than block elements
        result[i] = func(blocked_data[1:], axis=0)
    return result.T


def parametric_bootstrap(
    rvs: List[Callable],
    n_samples: int = 9999,
) -> npt.ArrayLike:
    """Resample data given a set of distributions

    Args:
        rvs (List[Callable]): list of random valuable generators
        n_samples (int, optional): Number of samples to generate of the set. Defaults to 9999.

    Returns:
        npt.ArrayLike: Array of results
    """
    res = np.empty((len(rvs), n_samples))

    for i, rv in enumerate(rvs):
        res[i] = rv(size=n_samples)

    return res
