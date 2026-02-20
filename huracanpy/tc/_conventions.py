"""
Module defining objects for TC-related conventions
"""

import numpy as np
from metpy.units import units

# %% Categories
_thresholds = {
    "Klotzbach": dict(
        bins=np.asarray([-np.inf, 925, 945, 960, 975, 990, 1005, np.inf])
        * units("hPa"),
        labels=[5, 4, 3, 2, 1, 0, -1],
    ),
    "Simpson": dict(
        bins=np.asarray([-np.inf, 920, 945, 965, 970, 980, 990, np.inf]) * units("hPa"),
        labels=[5, 4, 3, 2, 1, 0, -1],
    ),
    "10min": dict(
        bins=np.asarray([-np.inf, 16, 29, 38, 44, 52, 63, np.inf]) * units("m s-1"),
        labels=[-1, 0, 1, 2, 3, 4, 5],
    ),
    # https://www.nhc.noaa.gov/aboutsshws.php
    "1min": dict(
        bins=np.asarray([-np.inf, 34, 64, 83, 96, 113, 137, np.inf]) * units("kts"),
        labels=[-1, 0, 1, 2, 3, 4, 5],
    ),
}

# Aliases
_thresholds["Saffir-Simpson"] = _thresholds["10min"]
_thresholds["wmo"] = _thresholds["10min"]
_thresholds["nhc"] = _thresholds["1min"]
