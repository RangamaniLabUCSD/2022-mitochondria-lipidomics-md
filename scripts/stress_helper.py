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


# simulations = ["1", "2", "3", "4", "5", "6", "7"]
simulations = ["8", "9", "10", "11"]

def _stage(sim):

    # print(f"Processing {sim}...")
    sim_dir = util.sim_path / sim
    assert sim_dir.exists()

    ref_configuration = sim_dir / "production5+100.gro"
    if not ref_configuration.exists():
        return

    top = sim_dir / "system.top"
    assert top.exists()

    stress_mdp = util.mdp_path / "step8_stress.mdp"
    assert stress_mdp.exists()

    # Make sure the staging directory exists
    staging_dir = util.analysis_path / sim
    if not staging_dir.exists():
        os.makedirs(staging_dir)

    stresscalc_dir = staging_dir / "stress_calc"
    if not stresscalc_dir.exists():
        os.mkdir(stresscalc_dir)

    ##### GO INTO STAGING_DIR
    os.chdir(staging_dir)

    # PARSE INDEXES
    ndx = sim_dir / "index.ndx"
    # if ndx.exists():
    #     ndx.unlink()
    # subprocess.run(
    #     f'GMX_MAXBACKUP=-1 touch analysis.ndx; printf "a *\nname 0 System\nr POPC POPS POP2\nname 1 membrane\nq\nq" | {util.gmxls_bin} make_ndx -f {ref_configuration} -n analysis.ndx -o analysis.ndx',
    #     shell=True,
    #     check=True,
    #     capture_output=True,
    # )

    # check_ndx_cmd = f"{util.gmxls_bin} check -n analysis.ndx"
    # subprocess.run(check_ndx_cmd, shell=True, check=True, capture_output=True)

    if (stresscalc_dir / "stress.tpr").exists():
        (stresscalc_dir / "stress.tpr").unlink()

    # GROMPP generate stress.tpr
    grompp_cmd = f"{util.gmxls_bin} grompp -p {sim_dir}/system.top -f {stress_mdp} -n {ndx} -o ./stress_calc/stress.tpr -c {ref_configuration} -maxwarn 10"
    subprocess.run(grompp_cmd, shell=True, check=True, capture_output=True)

    system_index_cmd = f"{util.gmxls_bin} check -n {ndx} | grep System | awk '{{ print $1 }}'"
    p = subprocess.run(
        system_index_cmd, shell=True, check=True, capture_output=True
    )
    if not p.stdout:
        raise RuntimeError("Could not identify system index")
    system_index = int(p.stdout)

    membrane_index_cmd = f"{util.gmxls_bin} check -n {ndx} | grep membrane | awk '{{ print $1 }}'"
    p = subprocess.run(
        membrane_index_cmd, shell=True, check=True, capture_output=True
    )
    if not p.stdout:
        raise RuntimeError("Could not identify membrane index")
    membrane_index = int(p.stdout)

    #########################
    #  TRAJCONV
    #########################
    original_traj = sim_dir / "production5+100.trr"
    # centered_traj = staging_dir / "cen.trr"
    # # In theory it would be good to validate that the centered_traj is complete
    # # but there's no great way to do this without a lot of effort
    # if centered_traj.exists():
    #     print(f"{centered_traj}... exists")
    # else:
    #     trjconv_cmd = f"echo '{membrane_index} {system_index}' | {util.gmxls_bin} trjconv -f {original_traj} -o {centered_traj} -n {ndx} -center -s ./stress_calc/stress.tpr"
    #     # print(trjconv_cmd)
    #     subprocess.run(trjconv_cmd, shell=True, check=True)

    # if (staging_dir / "frames").exists():
    #     pass
    # else:
    # os.mkdir("frames")
    # SPLIT INTO REPRESENTATIVE FRAMES
    trjconv_cmd = f"echo '{membrane_index} {system_index}' | {util.gmxls_bin} trjconv -f {original_traj} -o ./frames/frame.trr -n {ndx} -center -split 5 -s ./stress_calc/stress.tpr"
    # print(trjconv_cmd)
    subprocess.run(trjconv_cmd, shell=True, check=True)


def stage():
    process_map(_stage, simulations, max_workers=7)



def _compute_stress(args):
    tpr = args[0]
    trr = args[1]
    out = args[2]
    cwd = args[3]

    os.chdir(cwd)
    rerun_cmd = f"GMX_MAXBACKUP=-1 {util.gmxls_bin} mdrun -s {tpr} -rerun {trr} -ols {out} -localsgrid 0.1 -lsgridx 27 -lsgridy 27"
    # print(rerun_cmd)
    p = subprocess.run(rerun_cmd, shell=True, check=True, capture_output=True)
    if p.returncode != 0:
        print(p.stderr)


def calculate_stresses():
    jobs = []

    # Iterate over simulations to process
    for sim in simulations:
        print(f"\tProcessing {sim}...")
        staging_dir = util.analysis_path / sim

        stresscalc_dir = staging_dir / "stress_calc"
        if not stresscalc_dir.exists():
            os.mkdir(stresscalc_dir)
        if not (stresscalc_dir / "frames").exists():
            os.mkdir(stresscalc_dir / "frames")

        if not (util.analysis_path / f"{sim}/frames").exists():
            print(f"Frames for system {sim} are missing... continuing...")
            continue

        tpr = staging_dir / "stress_calc/stress.tpr"
        assert tpr.exists()

        for i in range(0, 20001):
            frame = util.analysis_path / f"{sim}/frames/frame{i}.trr"
            if not frame.exists():
                continue
            frame_stress = stresscalc_dir / f"frames/frame{i}.dat"
            if (stresscalc_dir / f"frames/frame{i}.dat0").exists():
                continue
            jobs.append([tpr, frame, frame_stress, staging_dir])
    process_map(_compute_stress, jobs, max_workers=24, chunksize=1)


# def _postprocess(args):
#     stresscalc_dir = args[0]
#     files = args[1]

#     # print(f"Starting {stresscalc_dir}")
#     os.chdir(stresscalc_dir)

#     if not (stresscalc_dir / "averaged_stress.dat0").exists():
#         tensortools_cmd = f"python {util.script_path}/tensortools.py -f frames/frame*.dat0 -o {stresscalc_dir}/averaged_stress.dat0"
#         subprocess.run(tensortools_cmd, shell=True, check=True)

#     tensortools_cmd = f"python {util.script_path}/tensortools.py --prof z -f {stresscalc_dir}/averaged_stress.dat0 -o {stresscalc_dir}/z_profile_stress.txt"
#     subprocess.run(tensortools_cmd, shell=True, check=True)

#     arr = pd.read_csv(
#         f"{stresscalc_dir}/z_profile_stress.txt",
#         sep="\t",
#         header=None,
#         names=["z", "Sxx", "Sxy", "Sxz", "Syx", "Syy", "Syz", "Szx", "Szy", "Szz"],
#         skiprows=[0, 1],
#         usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     )

#     size = max(arr["z"])
#     arr["z"] = arr["z"].apply(
#         lambda x: x - (size / 2)
#     )  # adjusting so that zero is the midpoint, which should be the middle of the bilayer if the centering worked

#     lateral = -0.5 * 100 * (arr["Sxx"] + arr["Syy"])  # kpa
#     normal = -100 * arr["Szz"]  # kpa
#     lp = lateral - normal
#     arr["LP_(kPA)"] = lp
#     # print(lp)

#     stress = ["Sxx", "Sxy", "Sxz", "Syx", "Syy", "Syz", "Szx", "Szy", "Szz"]
#     arr[stress] = arr[stress].apply(lambda x: x * 100)  # converting the data into kPa
#     # print(arr['Sxx'])
#     arr.to_csv(f"{stresscalc_dir}/lateral_pressure.csv")


# def postprocess():
#     jobs = []

#     print(len(util.simulations))
#     # Iterate over simulations to process
#     for sim, t in util.simulations.items():
#         print(f"Processing {sim}...")
#         staging_dir = util.analysis_path / sim
#         stresscalc_dir = staging_dir / "stress_calc"

#         files = []
#         for i in range(0, 20001):
#             frame_stress = stresscalc_dir / f"frames/frame{i}.dat0"
#             if not frame_stress.exists():
#                 print(f"{frame_stress} is missing")
#             files.append(frame_stress)
#         jobs.append((stresscalc_dir, files))
#     process_map(_postprocess, jobs, max_workers=12)



def _z_profile_worker(args):
    stresscalc_dir = args[0]
    i = args[1]

    os.chdir(stresscalc_dir)

    # if not (stresscalc_dir / "averaged_stress.dat0").exists():
    tensortools_cmd = f"python {util.script_path}/tensortools.py --prof z -f frames/frame{i}.dat0 -o frames_z/frame_z_{i}.dat0"
    p = subprocess.run(tensortools_cmd, shell=True)
    if p.returncode != 0:
        print(p.stderr, stresscalc_dir, i)



def generate_z_profiles():
    print("Integrating to obtain z-profiles per frame...")
    jobs = []
    # Iterate over simulations to process
    for sim in simulations:
        print(f"\tScheduling jobs for {sim}...")
        staging_dir = util.analysis_path / sim
        stresscalc_dir = staging_dir / "stress_calc"
        
        frames_z_dir = stresscalc_dir / "frames_z"

        if not frames_z_dir.exists():
            frames_z_dir.mkdir()

        for i in range(0, 20001):
            frame_stress = stresscalc_dir / f"frames/frame{i}.dat0"
            if not frame_stress.exists():
                print(f"{frame_stress} is missing")
                continue
            if not (frames_z_dir / f"frame_z_{i}.dat0").exists():
                jobs.append((stresscalc_dir, i))
    process_map(_z_profile_worker, jobs, max_workers=24, chunksize=100)


def _average_z_worker(stresscalc_dir):
    tensortools_cmd = f"python {util.script_path}/tensortools.py -f {stresscalc_dir}/frames_z/frame_z_*.dat0 -o {stresscalc_dir}/z_profile_stress.txt"
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

def average_z_profiles():
    print("Averaging z-profiles and generating lateral pressure profile...")
    jobs = []

    # Iterate over simulations to process
    for sim in simulations:
        print(f"\tScheduling jobs for {sim}...")
        staging_dir = util.analysis_path / sim
        stresscalc_dir = staging_dir / "stress_calc"

        if not (stresscalc_dir / "frames_z").exists():
            print(f"\t\t Missing frames_z dir at {stresscalc_dir}")
            continue
        else:
            for i in range(0, 20001):
                if not (stresscalc_dir / f"frames_z/frame_z_{i}.dat0").exists():
                    print(f"\t\t{stresscalc_dir / f'frames_z/frame_z_{i}.dat0'} is missing...")
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

    if options.stage:
        stage()

    if options.run:
        calculate_stresses()

    if options.postprocess:
        generate_z_profiles()
        average_z_profiles()
