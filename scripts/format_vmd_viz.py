from string import Template
from pathlib import Path
import subprocess
import os

from tqdm.contrib.concurrent import process_map

import util


types = ["side", "top"]

def render_job(args):
    (cwd, type) = args
    os.chdir(cwd)
    subprocess.call(f"vmd -e viz_{type}.vmd", shell=True)


if __name__ == "__main__":
    jobs = []

    for type in types:
        template_file = Path(f"VMD_VIZ_{type}.template")

        if template_file.exists():
            with template_file.open("r") as fd:
                src = Template(fd.read())
        else:
            raise RuntimeError(f"Could not find template file: {template_file}")


        for sim in range(1, 25):
            renderPath = util.analysis_fast_path / f"{sim}/render_{type}"
            renderPath.mkdir(parents=True, exist_ok=True)

            jobs.append((util.analysis_fast_path / f"{sim}", type))

            # print(sim)
            d = {
                "SIM": str(sim),
                "PATH": str(util.analysis_path),
            }

            result = src.substitute(d)
            vizPath = util.analysis_fast_path / f"{sim}/viz_{type}.vmd"

            with vizPath.open(mode="w") as fd:
                fd.write(result)
                for i, frame in enumerate(range(0, 10001, 10)):
                    fd.write(f"animate goto {frame}\n")
                    fd.write(
                        f"render TachyonInternal {renderPath}/System{util.sim_to_final_index[sim]}_{type}_frame{i:05d}.ppm\n"
                    )

                fd.write(f"mogrify -format png {renderPath}/*.ppm\n")
                fd.write(f"rm {renderPath}/*.ppm\n")
                fd.write("exit\n")
    process_map(render_job, jobs, max_workers=12)
