# script to make copies of enth domains

import MDAnalysis as mda

from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from pathlib import Path

import os
import util
import subprocess



# simulations = ["1", "2", "3", "4", "5", "6", "7"]
simulations = ["8", "9", "10", "11"]

def _strip_trajectory(sim):
    staging_dir = util.analysis_path / sim
    os.chdir(staging_dir)

    trjconv_cmd = f"echo '1' | {util.gmxls_bin} trjconv -f {util.sim_path}/{sim}/production5+100.gro -o po4_only.gro -n po4_membrane.ndx -s analysis.tpr"
    subprocess.run(trjconv_cmd, shell=True, check=True)

    for i in range(1,6):
        trjconv_cmd = f"echo '0 1' | {util.gmxls_bin} trjconv -f {util.sim_path}/{sim}/production{i}.trr -center -o po4_{i}.xtc -n {util.analysis_path / sim}/po4_membrane.ndx -s {util.analysis_path}/{sim}/analysis.tpr"
        # print(trjconv_cmd)
        subprocess.run(trjconv_cmd, shell=True, check=True)


jobs = []
for sim in tqdm(simulations, desc=f"Generating ndx"):
    curr_dir = util.analysis_path / sim

    u = mda.Universe(
        f"{util.sim_path}/{sim}/system.top",
        f"{util.sim_path}/{sim}/production5+100.gro",
        topology_format="ITP",
    )
    membrane = u.atoms.select_atoms("resname POPC DOPC POPE DOPE CDL1 POPG DOPG")
    po4 = u.atoms.select_atoms("name PO4 or name PO41 or name PO42 or name GL0")

    # print(len(po4))

    with mda.selections.gromacs.SelectionWriter(
        f"{curr_dir}/po4_membrane.ndx", mode="w"
    ) as ndx:
        ndx.write(membrane, name="membrane")
        ndx.write(po4, name="po4")

    # Generate analysis.tpr
    cmd = f"{util.gmxls_bin} grompp -p {util.sim_path}/{sim}/system.top -f {util.mdp_path}/step7.2_production.mdp -n {util.sim_path / sim}/index.ndx -maxwarn 10 -c {util.sim_path}/{sim}/equilibration4.gro -o {util.analysis_path}/{sim}/analysis.tpr"
    subprocess.run(cmd, shell=True, check=True)

    jobs.append(sim)

r = process_map(_strip_trajectory, jobs, max_workers=12)
