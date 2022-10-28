#!/usr/bin/env python

"""Center and extract trajectories of headgroup beads.
"""
import MDAnalysis
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import os
import subprocess

import util # Local utilities and definitions

def _strip_trajectory(sim: str) -> None:
    """Helper function to run trajconv to center and produce stripped trajectories

    Args:
        sim (str): Name of the system
    """

    source_dir = util.sim_path / sim
    staging_dir = util.analysis_path / sim
    staging_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(staging_dir)

    trjconv_cmd = f"echo '1' | {util.gmxls_bin} trjconv -f {source_dir}/production5+100.gro -o po4_only.gro -n po4_membrane.ndx -s analysis.tpr"
    subprocess.run(trjconv_cmd, shell=True, check=True)

    for i in range(1, 6):
        trjconv_cmd = f"echo '0 1' | {util.gmxls_bin} trjconv -f {source_dir}/production{i}.trr -center -o po4_{i}.xtc -n {staging_dir}/po4_membrane.ndx -s {staging_dir}/analysis.tpr"
        # print(trjconv_cmd)
        subprocess.run(trjconv_cmd, shell=True, check=True)


# Generate index files and process trajectories in parallel
jobs = []
for sim in tqdm(util.simulations, desc=f"Generating ndx and analysis.tpr"):
    source_dir = util.sim_path / sim
    staging_dir = util.analysis_path / sim
    staging_dir.mkdir(parents=True, exist_ok=True)

    u = MDAnalysis.Universe(
        f"{source_dir}/system.top",
        f"{source_dir}/production5+100.gro",
        topology_format="ITP",
    )
    membrane = u.atoms.select_atoms(util.membrane_sel)
    po4 = u.atoms.select_atoms(util.po4_sel)

    with MDAnalysis.selections.gromacs.SelectionWriter(
        f"{staging_dir}/po4_membrane.ndx", mode="w"
    ) as ndx:
        ndx.write(membrane, name="membrane")
        ndx.write(po4, name="po4")

    # Generate analysis.tpr
    cmd = f"{util.gmxls_bin} grompp -p {source_dir}/system.top -f {util.mdp_path}/step7.2_production.mdp -n {source_dir}/index.ndx -maxwarn 10 -c {source_dir}/equilibration4.gro -o {staging_dir}/analysis.tpr"
    subprocess.run(cmd, shell=True, check=True)

    mdout_mdp = Path.cwd() / "mdout.mdp"
    if mdout_mdp.exists():
        mdout_mdp.unlink()

    jobs.append(sim)

r = process_map(_strip_trajectory, jobs, max_workers=12)
