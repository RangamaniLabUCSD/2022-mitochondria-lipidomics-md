from pathlib import Path

import util

import shutil


for sim in util.simulations:
    top = util.sim_path  / f"{sim}/system.top"

    new_top = util.analysis_fast_path / f"{sim}/system.top"

    with open(new_top, "w") as newfd:
        with open(top, 'r') as fd:
            for line in fd.readlines():
                if line.startswith('#include "../'):
                    line = line.replace('#include "../', '#include "/scratch2/ctlee/mito_lipidomics_scratch2/analysis/')
                if line.startswith("W "):
                    line = ";" + line
                elif line.startswith("NA "):
                    line = ";" + line
                elif line.startswith("CL "):
                    line = ";" + line
                print(line)
                newfd.write(line)
                
