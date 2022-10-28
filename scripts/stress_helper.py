#!/usr/bin/env python

"""
Scripts to copy and run stress calculations
"""

from tqdm.contrib.concurrent import process_map
import os
import subprocess
import util
import argparse

from LStensor import LStensor

import pandas as pd


def _stage(sim):
    """Stage trajectories for parallel stress calculation

    Args:
        sim (str): Simulation directory
    """
    print(sim)
    sim_dir = util.scratch_sim_path / sim
    if not sim_dir.exists():
        raise RuntimeError(f"{sim_dir} is missing")

    ref_configuration = sim_dir / "production3+100.gro"
    if not ref_configuration.exists():
        return

    top = sim_dir / "system.top"
    assert top.exists()
    ndx = sim_dir / "index.ndx"
    assert ndx.exists()

    stress_mdp = util.mdp_path / "step8_stress.mdp"
    assert stress_mdp.exists()

    # Make sure the staging directory exists
    staging_dir = util.analysis_path / sim
    stresscalc_dir = staging_dir / "stress_calc"
    stresscalc_dir.mkdir(parents=True, exist_ok=True)

    ##### GO INTO STAGING_DIR
    os.chdir(staging_dir)

    ### GROMPP generate stress.tpr
    if (stresscalc_dir / "stress.tpr").exists():
        (stresscalc_dir / "stress.tpr").unlink()

    grompp_cmd = f"{util.gmxls_bin} grompp -p {top} -f {stress_mdp} -n {ndx} -o ./stress_calc/stress.tpr -c {ref_configuration} -maxwarn 10"
    subprocess.run(grompp_cmd, shell=True, check=True, capture_output=True)

    ### GET NUMBERS CORRESPONDING TO INDEXES
    system_index_cmd = (
        f"{util.gmxls_bin} check -n {ndx} | grep system | awk '{{ print $1 }}'"
    )
    p = subprocess.run(system_index_cmd, shell=True, check=True, capture_output=True)
    if not p.stdout:
        raise RuntimeError("Could not identify system index")
    system_index = int(p.stdout)

    membrane_index_cmd = (
        f"{util.gmxls_bin} check -n {ndx} | grep membrane | awk '{{ print $1 }}'"
    )
    p = subprocess.run(membrane_index_cmd, shell=True, check=True, capture_output=True)
    if not p.stdout:
        raise RuntimeError("Could not identify membrane index")
    membrane_index = int(p.stdout)

    #### RUN TRAJCONV
    original_traj = sim_dir / "production3+100.trr"
    (staging_dir / "frames").mkdir(parents=True, exist_ok=True)

    trjconv_cmd = f"echo '{membrane_index} {system_index}' | {util.gmxls_bin} trjconv -f {original_traj} -o {staging_dir}/frames/frame.trr -n {ndx} -center -split 5 -s ./stress_calc/stress.tpr"
    subprocess.run(trjconv_cmd, shell=True, check=True)


def stage(jobs):
    """Center and split trajectories for fast parallel calculation of stress"""
    process_map(_stage, jobs, max_workers=7)


def _compute_stress(args):
    """Compute stresses by rerunning

    Args:
        args (Tuple): Tuple of TPR, TRR, OUT, CWD
    """
    tpr, trr, out, cwd = args
    os.chdir(cwd)
    rerun_cmd = f"GMX_MAXBACKUP=-1 {util.gmxls_bin} mdrun -s {tpr} -rerun {trr} -ols {out} -localsgrid 0.1 -lsgridx 1 -lsgridy 1"
    p = subprocess.run(rerun_cmd, shell=True, check=True, capture_output=True)
    if p.returncode != 0:
        print(p.stderr)


def calculate_stresses(sims):
    jobs = []

    # Iterate over simulations to process
    for sim in sims:
        print(f"\tProcessing {sim}...")
        staging_dir = util.analysis_path / f"{sim}"

        stresscalc_dir = staging_dir / "stress_calc"
        (stresscalc_dir / "frames").mkdir(parents=True, exist_ok=True)

        if not (staging_dir / "frames").exists():
            print(f"Frames for system {sim} are missing... continuing...")
            continue

        tpr = stresscalc_dir / "stress.tpr"
        assert tpr.exists()

        # TODO: change this to not be a hardcoded number
        for i in range(0, 20001):
            frame = staging_dir / f"frames/frame{i}.trr"
            if not frame.exists():
                print(f"Missing frame: {frame}")
                continue
            frame_stress = stresscalc_dir / f"frames/frame{i}.dat"
            if (stresscalc_dir / f"frames/frame{i}.dat0").exists():
                continue
            jobs.append((tpr, frame, frame_stress, staging_dir))
    process_map(_compute_stress, jobs, max_workers=24, chunksize=100)


# def _z_profile_worker(args):
#     stresscalc_dir, i = args
#     os.chdir(stresscalc_dir)

#     tensortools_cmd = f"python {util.script_path}/tensortools.py --prof z -f frames/frame{i}.dat0 -o frames_z/frame_z_{i}.dat0"
#     p = subprocess.run(tensortools_cmd, shell=True)
#     if p.returncode != 0:
#         print(p.stderr, stresscalc_dir, i)


# def generate_z_profiles(sims):
#     print("Integrating to obtain z-profiles per frame...")
#     jobs = []
#     # Iterate over simulations to process
#     for sim in sims:
#         print(f"\tScheduling jobs for {sim}...")
#         stresscalc_dir = util.analysis_path / sim / "stress_calc"
#         frames_z_dir = stresscalc_dir / "frames_z"
#         frames_z_dir.mkdir(exist_ok=True)

#         # TODO: change this to not be a hardcoded number
#         for i in range(0, 20001):
#             frame_stress = stresscalc_dir / f"frames/frame{i}.dat0"
#             if not frame_stress.exists():
#                 print(f"{frame_stress} is missing")
#                 continue
#             if not (frames_z_dir / f"frame_z_{i}.dat0").exists():
#                 jobs.append((stresscalc_dir, i))
#     process_map(_z_profile_worker, jobs, max_workers=24, chunksize=100)


def _average_z_worker(stresscalc_dir):
    tensortools_cmd = f"python {util.script_path}/tensortools.py -f {stresscalc_dir}/frames/frame*.dat0 -o {stresscalc_dir}/z_profile_stress.txt"
    subprocess.run(tensortools_cmd, shell=True, check=True)

    arr = pd.read_csv(
        f"{stresscalc_dir}/z_profile_stress.txt",
        sep="\t",
        header=None,
        names=["z", "Sxx", "Sxy", "Sxz", "Syx", "Syy", "Syz", "Szx", "Szy", "Szz"],
        skiprows=[0, 1],
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    size = max(arr["z"])
    arr["z"] = arr["z"].apply(
        lambda x: x - (size / 2)
    )  # adjusting so that zero is the midpoint, which should be the middle of the bilayer if the centering worked

    lateral = -0.5 * 100 * (arr["Sxx"] + arr["Syy"])  # kpa
    normal = -100 * arr["Szz"]  # kpa
    lp = lateral - normal
    arr["LP_(kPA)"] = lp
    # print(lp)

    stress = ["Sxx", "Sxy", "Sxz", "Syx", "Syy", "Syz", "Szx", "Szy", "Szz"]
    arr[stress] = arr[stress].apply(lambda x: x * 100)  # converting the data into kPa
    # print(arr['Sxx'])
    arr.to_csv(f"{stresscalc_dir}/lateral_pressure.csv")


def average_z_profiles(sims):
    print("Averaging z-profiles and generating lateral pressure profile...")
    jobs = []

    # Iterate over simulations to process
    for sim in sims:
        print(f"\tScheduling jobs for {sim}...")
        stresscalc_dir = util.analysis_path / sim / "stress_calc"

        if not (stresscalc_dir / "frames").exists():
            print(f"\t\t Missing frames dir at {stresscalc_dir}")
            continue
        else:
            for i in range(0, 20001):
                if not (stresscalc_dir / f"frames/frame{i}.dat0").exists():
                    print(
                        f"\t\t{stresscalc_dir / f'frames/frame{i}.dat0'} is missing..."
                    )
                    break
        jobs.append((stresscalc_dir))
    process_map(_average_z_worker, jobs, max_workers=12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--stage",
        action="store_true",
        help="Stage and center trajectories for stress calculation",
    )
    parser.add_argument(
        "-r", "--run", action="store_true", help="Run stress calculation per frame"
    )
    parser.add_argument("-p", "--postprocess", action="store_true", help="Post process")

    options = parser.parse_args()

    jobs = []
    for sim in util.simulations:
        jobs.append(f"{sim}_small")

    if options.stage:
        stage(jobs)

    if options.run:
        calculate_stresses(jobs)

    if options.postprocess:
        # generate_z_profiles(jobs)
        average_z_profiles(jobs)
