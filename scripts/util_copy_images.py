from string import Template
from pathlib import Path
import subprocess
import os
import shutil

from tqdm.contrib.concurrent import process_map

import util

types = ["side", "top"]

sims = [4, 5, 6, 1, 9, 7]
i = 853
if __name__ == "__main__":
    for type in types:
        for sim in sims:
            renderPath = (
                util.analysis_fast_path / f"{util.remapping_dict[sim]}/render_{type}"
            )
            img = renderPath / f"System{sim}_{type}_frame{i:05d}.png"
            if not img.exists():
                print(img, "does not exist")
        
            shutil.copy(img, ".")
            
