import pickle
import numpy as np
from pathlib import Path
import util
from LStensor import LStensor

from tqdm.auto import tqdm

# Override and recompute even if spectra pickle exists
lp_compute_override = True

lp_fd = util.analysis_path / "lp.pickle"


if lp_fd.exists() and not lp_compute_override:
    # LOAD LP pickle
    with open(lp_fd, "rb") as handle:
        lateral_pressure = pickle.load(handle)
    print("Loaded LP from cache")

else:
    lateral_pressure = {}

    for sim in tqdm(util.simulations, position=0):
        fd = Path(util.analysis_path /  f"{sim}_small/stress_calc/frames/frame0.dat0")
        field = LStensor(2)
        field.g_loaddata(files=[fd], bAvg="avg")

        # stress_tensor = np.empty((20000, field.nz, 9))
        lateral_pressure[sim] = np.empty((40000, field.nz, 3))

        # 0-20000 frames in each trajectory
        for i, j in tqdm(enumerate(range(1, 40001)), total=40000, position=1):
            fd = Path(util.analysis_path / f"{sim}_small/stress_calc/frames/frame{j}.dat0")
            field = LStensor(2)
            field.g_loaddata(files=[fd], bAvg="avg")
            stress_tensor = field.data_grid * 100   # Convert to kPa from 10^5 Pa
            # Sxx Sxy Sxz Syx Syy Syz Szx Szy Szz
            # 0               4               8

            pXY = -0.5*(stress_tensor[:,0] + stress_tensor[:,4]).reshape(-1,1)
            pN = (-stress_tensor[:,8]).reshape(-1,1)
            lp = pXY - pN
            z = (np.arange(field.nz) * field.dz - (field.nz - 1) * field.dz / 2).reshape(-1,1)
            lateral_pressure[sim][i] = np.hstack((pN, lp, z))
    
    # WRITE LP TO PICKLE
    with open(lp_fd, "wb") as handle:
        pickle.dump(lateral_pressure, handle, protocol=pickle.HIGHEST_PROTOCOL)
