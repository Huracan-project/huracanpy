"""
Module defining objects for TC-related conventions
"""

import numpy as np
from metpy.units import units

# %% Categories
_thresholds = {
    "Klotzbach": dict(
        bins=np.array([-np.inf, 925, 945, 960, 975, 990, 1005, np.inf]) * units("hPa"),
        labels=[5, 4, 3, 2, 1, 0, -1],
    ),
    "Simpson": dict(
        bins=np.array([-np.inf, 920, 945, 965, 970, 980, 990, np.inf]) * units("hPa"),
        labels=[5, 4, 3, 2, 1, 0, -1],
    ),
    "Saffir-Simpson": dict(
        bins=np.array([-np.inf, 16, 29, 38, 44, 52, 63, np.inf]) * units("m s-1"),
        labels=[-1, 0, 1, 2, 3, 4, 5],
    ),
}
