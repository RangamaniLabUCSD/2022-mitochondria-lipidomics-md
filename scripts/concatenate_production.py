#!/usr/bin/env python

"""Center and extract trajectories of headgroup beads.
"""
import MDAnalysis
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import os
import subprocess

import util  # Local utilities and definitions


def _concatenate_trajectory(sim: str) -> None:
    """Helper function to run trajconv to center and produce stripped trajectories

    Args:
        sim (str): Name of the system
    """
    source_dir = util.sim_path / sim
    os.chdir(source_dir)

    # subprocess.run("gmx trjcat -f production.part0002.trr production.part0003.trr production.part0004.trr production.part0005.trr production.trr -o production_all.trr", shell=True, check=True)

    subprocess.run("gmx trjcat -f production+100*.trr -o production_all.trr", shell=True, check=True)


# Generate index files and process trajectories in parallel
jobs = []
for sim in tqdm(util.simulations, desc=f"Generating ndx and analysis.tpr"):
    # sim = f"{sim}_small"
    source_dir = util.sim_path / str(sim)
    staging_dir = util.analysis_path / str(sim)
    staging_dir.mkdir(parents=True, exist_ok=True)

    if (source_dir / "production_all.trr").exists():
       print(source_dir / "production_all.trr", "EXISTS")
       continue
    jobs.append(sim)

r = process_map(_concatenate_trajectory, jobs, max_workers=2)
