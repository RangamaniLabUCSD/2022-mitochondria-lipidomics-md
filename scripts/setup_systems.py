#!/usr/bin/env python

import subprocess

import MDAnalysis
import os
from pathlib import Path

import util

large_box_size = [40, 40, 25]
small_box_size = [15, 15, 25]

# Check that mol fractions add up to 100%
for sim in util.simulations:
    v = 0
    for lipid, chi in util.simulations[sim].items():
        v += chi
    if v != 100:
        raise RuntimeError(f"Molfractions for system {sim} don't add up to 100")

sim_path = Path("sims_new")

for sim, composition in util.simulations.items():
    for size in ["large", "small"]:
        if size == "large":
            box_size = large_box_size
            sim_dir = sim_path / f"{sim}"
        else:
            box_size = small_box_size
            sim_dir = sim_path / f"{sim}_small"

        topfile = sim_dir / "system.top"
        grofile = sim_dir / "initial.gro"

        sim_dir.mkdir(parents=True, exist_ok=True)

        if size == "small":
            args = (
                args
            ) = f"insane -salt 0.15 -sol W:90 -sol WF:10 -x {box_size[0]} -y {box_size[1]} -z {box_size[2]} -o {grofile} -p {topfile}  -pbc rectangular "
        else:
            args = f"insane -salt 0.15 -sol W -x {box_size[0]} -y {box_size[1]} -z {box_size[2]} -o {grofile} -p {topfile}  -pbc rectangular "
        for lipid, chi in composition.items():
            if chi == 0:
                continue
            args += f"-l {lipid}:{chi} -u {lipid}:{chi} "
        # print(args)
        subprocess.run(args, shell=True, check=True)

        u = MDAnalysis.Universe(topfile, grofile, topology_format="ITP")
        # print(u)

        membrane = u.atoms.select_atoms(util.membrane_sel)

        with MDAnalysis.selections.gromacs.SelectionWriter(
            f"{sim_dir}/index.ndx", mode="w"
        ) as ndx:
            ndx.write(u.atoms, name="system")
            ndx.write(membrane, name="membrane")


# Compare the discretized composition to the target composition
for sim, composition in util.simulations.items():
    for size in ["large", "small"]:
        if size == "large":
            box_size = large_box_size
            sim_dir = sim_path / f"{sim}"
        else:
            box_size = small_box_size
            sim_dir = sim_path / f"{sim}_small"

        topfile = sim_dir / "system.top"
        grofile = sim_dir / "initial.gro"

        u = MDAnalysis.Universe(topfile, grofile, topology_format="ITP")
        d = util.count_residues(u)

        total_lipids = 0
        for k in util.lipid_names:
            if k in d:
                total_lipids += d[k]

        normed_comp = {}
        discretized_composition_str = ""
        base_composition_str = ""
        for k in util.lipid_names:
            if k in d:
                discretized_composition_str += f"{k}: {100*d[k]/total_lipids:0.2f}; "
                base_composition_str += f"{k}: {composition[k]:0.2f}; "
                normed_comp[k] = d[k] / total_lipids
            else:
                discretized_composition_str += f"{k}: {0:0.2f}; "
                base_composition_str += f"{k}: {0:0.2f}; "
                normed_comp[k] = 0
        print(sim_dir)
        print(discretized_composition_str)
        print(base_composition_str)
        print()
