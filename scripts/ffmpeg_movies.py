from pathlib import Path
import util

import os
import subprocess

base_path = util.script_path / "render_movies"

for sim in util.simulations:
    sim_name = f"{util.sim_to_final_index[sim]}"

    cwd = base_path / sim_name
    os.chdir(cwd)


    #### HEVC x265
    # subprocess.run(
    #     f"ffmpeg -y -r 30 -pattern_type glob -i 'mean/frame_mean_0*.png' -vf format=yuv444p10le -an -c:v libx265 -crf 25 -tag:v hvc1 sys{sim_name}_mean.mp4",
    #     shell=True,
    # )
    # subprocess.run(
    #     f"ffmpeg -y -r 30 -pattern_type glob -i 'mean/frame_mean_dark_*.png' -vf format=yuv444p10le -an -c:v libx265 -crf 25 -tag:v hvc1 sys{sim_name}_mean_dark.mp4",
    #     shell=True,
    # )

    # subprocess.run(
    #     f"ffmpeg -y -r 30 -pattern_type glob -i 'height/frame_height_0*.png' -vf format=yuv444p10le -an -c:v libx265 -crf 25 -tag:v hvc1 sys{sim_name}_height.mp4",
    #     shell=True,
    # )

    subprocess.run(
        f"ffmpeg -y -r 30 -pattern_type glob -i 'height/frame_height_dark_*.png' -vf format=yuv444p10le -an -c:v libx265 -crf 25 -tag:v hvc1 sys{sim_name}_height_dark.mp4",
        shell=True,
    )


    ##### AV1 encoding (is not supported by quicktime)
    # subprocess.run(
    #     f"ffmpeg -y -r 30 -pattern_type glob -i 'mean/frame_mean_0*.png' -an -c:v libsvtav1 -crf 20 sys{sim_name}_mean.mp4",
    #     shell=True,
    # )
    # subprocess.run(
    #     f"ffmpeg -y -r 30 -pattern_type glob -i 'mean/frame_mean_dark_*.png' -an -c:v libsvtav1 -crf 20 sys{sim_name}_mean_dark.mp4",
    #     shell=True,
    # )
