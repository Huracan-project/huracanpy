"""
Auxilliary file to define the basins according to the different conventions
"""

from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd

basins_def = {}  # Dictionnary to save basin definition from different conventions


# WMO convention
## Northern hemisphere
NATL = Polygon(((260, 90), (360, 90), (360, 0), (295, 0), (260, 20)))
ENP = Polygon(((220, 90), (260, 90), (260, 20), (295, 0), (220, 0)))
CP = Polygon(((180, 0), (180, 90), (220, 90), (220, 0)))
WNP = Polygon(((100, 0), (100, 90), (180, 90), (180, 0)))
NI = Polygon(((30, 0), (30, 90), (100, 90), (100, 0)))
MED = Polygon(((0, 0), (0, 90), (30, 90), (30, 0)))
NH = {"NATL": NATL, "ENP": ENP, "CP": CP, "WNP": WNP, "NI": NI, "MED": MED}

## Southern hemisphere
SI = Polygon(((20, -90), (20, 0), (90, 0), (90, -90)))
AUS = Polygon(((90, -90), (90, 0), (160, 0), (160, -90)))
SP = Polygon(((160, 0), (160, -90), (295, -90), (295, 0)))
SA1 = Polygon(((295, -90), (295, 0), (360, 0), (360, -90)))
SA2 = Polygon(((20, -90), (20, 0), (0, 0), (0, -90)))
SA = MultiPolygon([SA1, SA2])
SA_plot = Polygon(((295, -90), (295, 0), (380, 0), (380, -90)))
SH = {"SI": SI, "AUS": AUS, "SP": SP, "SA": SA}

B = dict(SH, **NH)
basins_def["WMO"] = gpd.GeoDataFrame(index=B.keys(), geometry=list(B.values()))
