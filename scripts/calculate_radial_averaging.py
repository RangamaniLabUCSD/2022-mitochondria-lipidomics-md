import pickle
import numpy as np
from functools import partial
import MDAnalysis

from pathlib import Path

import matplotlib.pyplot as plt
import numpy.typing as npt

import pandas as pd

from scipy import integrate, interpolate, stats
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

import util

def radial_averaging_series(power2D, mc, min_bin=0.001, max_bin=1, bin_width=0.001):
    """
    Perform radial averaging over multiple frames in a time series. 

    Radially average the power spectrum to obtain values. Notably the natural freqeuncy unit
    of this function is A^-1.

    Args:
        power2D (numpy.array((M,N,N))): Power spectrum
        mc (_type_): Membrane curvature object with metadata
        min_bin (float, optional): Minimum bin value. Defaults to 0.001.
        max_bin (int, optional): Maximum bin value. Defaults to 1.
        bin_width (float, optional): Bin width. Defaults to 0.001.

    Returns:
        tuple: Binned power spectra
    """

    if not len(power2D.shape) == 3:
        raise RuntimeError("Expected time series of 2D power")

    x, y = np.meshgrid(mc["qx"], mc["qy"])  # A^-1
    r = np.sqrt(x**2 + y**2)
    bins = np.arange(min_bin, max_bin, bin_width)

    digitized = np.digitize(r, bins)
    bc = np.array(
        [
            r[digitized == i].mean() if np.count_nonzero(digitized == i) else np.NAN
            for i in range(1, len(bins))
        ]
    )

    first_iter = True

    spectra = None

    for i, frame in tqdm(enumerate(power2D), total=len(power2D)):
        bm = np.array(
            [
                frame[digitized == i].mean()
                if np.count_nonzero(digitized == i)
                else np.NAN
                for i in range(1, len(bins))
            ]
        )

        if i == 0:
            bin_centers = bc[np.isfinite(bm)]
            bin_means = bm[np.isfinite(bm)]
            spectra = np.zeros((power2D.shape[0], len(bin_means)))
            spectra[i] = bin_means
        else:
            spectra[i] = bm[np.isfinite(bm)]
    return (bin_centers, spectra)


mc = {}
for sim in util.simulations:
    with open(util.analysis_path / f"{sim}/membrane_curvature_2nm.pickle", "rb") as handle:
        mc[sim] = pickle.load(handle) 

####################################################
#### Strip out difficult to port objects from cache
####################################################
without_mdanalysis_objects = {}

skip = ["ag", "results", "run"]

for sim in util.simulations:
    without_mdanalysis_objects[sim] = {}

    for item in dir(mc[sim]):
        if not item.startswith("_") and item not in skip:
            # print(item, type(getattr(mc["1"],item)))
            without_mdanalysis_objects[sim][item] = getattr(mc[sim],item)
        elif item == "results":
            for key in mc[sim].results.keys():
                without_mdanalysis_objects[sim][key] = getattr(mc[sim],item)[key]

with open("mc_noobject_2nm.pickle", "wb") as handle:
    pickle.dump(without_mdanalysis_objects, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open("mc_noobject_2nm.pickle", "rb") as handle:
    mc = pickle.load(handle)

# Override and recompute even if spectra pickle exists
spectra_compute_override = True

spectra_fd = util.analysis_path / "spectra_2nm.pickle"
if spectra_fd.exists() and not spectra_compute_override:
    # LOAD SPECTRA PICKLE
    with open(spectra_fd, "rb") as handle:
        spectra = pickle.load(handle)
    print("Loaded spectra from cache!")
else:
    def compute_spectra(sim):
        return sim, radial_averaging_series(
            mc[sim]["height_power_spectrum"],
            mc[sim],
            min_bin=0.001,
            max_bin=1,
            bin_width=0.001,
        )

    spectra = dict(map(compute_spectra, util.simulations))

    # WRITE SPECTRA TO PICKLE
    with open(spectra_fd, "wb") as handle:
        pickle.dump(spectra, handle, protocol=pickle.HIGHEST_PROTOCOL)
