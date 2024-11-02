"""
Module defining objects for TC-related conventions
"""

from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd

import numpy as np
from metpy.units import units


# %% Basins

tc_basins = {}  # Dictionnary to save basin definition from different conventions

# WMO convention
## Northern hemisphere
NATL = Polygon(((-100, 90), (0, 90), (0, 0), (-65, 0), (-100, 20)))
ENP = Polygon(((-140, 90), (-100, 90), (-100, 20), (-65, 0), (-140, 0)))
CP = Polygon(((-180, 0), (-180, 90), (-140, 90), (-140, 0)))
WNP = Polygon(((100, 0), (100, 90), (180, 90), (180, 0)))
NI = Polygon(((30, 0), (30, 90), (100, 90), (100, 0)))
MED = Polygon(((0, 0), (0, 90), (30, 90), (30, 0)))
NH = {"NATL": NATL, "ENP": ENP, "CP": CP, "WNP": WNP, "NI": NI, "MED": MED}

## Southern hemisphere
SI = Polygon(((20, -90), (20, 0), (90, 0), (90, -90)))
AUS = Polygon(((90, -90), (90, 0), (160, 0), (160, -90)))
SP = MultiPolygon(
    [
        Polygon([(160, 0), (160, -90), (180, -90), (180, 0)]),
        Polygon(((-180, 0), (-180, -90), (-65, -90), (-65, 0))),
    ]
)
SA = Polygon(((-65, -90), (-65, 0), (20, 0), (20, -90)))
SH = {"SI": SI, "AUS": AUS, "SP": SP, "SA": SA}

B = dict(SH, **NH)
tc_basins["WMO-TC"] = gpd.GeoDataFrame(index=B.keys(), geometry=list(B.values()))

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
