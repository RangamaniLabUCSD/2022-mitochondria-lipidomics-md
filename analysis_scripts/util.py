import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder, optimize_cutoff

import re
from pathlib import Path

import math


# source_path = Path("/u1/ctlee/enth/system_building")
base_path = Path("/net/engram/ctlee/mito_lipidomics")

scratch_path = Path("/scratch/ctlee/mito_lipidomics_scratch")

sim_path = base_path / "sims"
scratch_sim_path = scratch_path / "sims"

mdp_path = base_path / "mdps"
script_path = base_path / "scripts"

analysis_path = scratch_path / "analysis"
analysis_base_path = base_path / "analysis"
analysis_special = Path("/u1/ctlee/mito_lipidomics")

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


# xsync 0enth_na [4-9]x[4-9]* clee2@tscc:~/lustre/enth/
