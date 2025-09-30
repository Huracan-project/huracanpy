from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd

# %% Basins

basins = {}  # Dictionnary to save basin definition from different conventions

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
basins["WMO-TC"] = gpd.GeoDataFrame(index=B.keys(), geometry=list(B.values()))

# Sainsbury et. al. (2022)
# What Governs the Interannual Variability of Recurving North Atlantic Tropical
# Cyclones?
# https://doi.org/10.1175/JCLI-D-21-0712.1
B = dict(
    MDR=Polygon([(-70, 6), (-10, 6), (-10, 20), (-70, 20)]),
    SUB=Polygon([(-82, 20), (-10, 20), (-10, 50), (-82, 50)]),
    WEST=Polygon(
        [
            (-70, 8),
            (-90, 8),
            (-90, 16),
            (-100, 16),
            (-100, 33),
            (-82, 33),
            (-82, 20),
            (-70, 20),
        ]
    ),
)
basins["Sainsbury2022JCLI"] = gpd.GeoDataFrame(
    index=B.keys(), geometry=list(B.values())
)

# Sainsbury et. al. (2022)
# Why Do Some Post-Tropical Cyclones Impact Europe?
# https://doi.org/10.1175/MWR-D-22-0111.1
B = dict(
    Europe=Polygon([(-10, 36), (30, 36), (30, 70), (-10, 70)]),
    NoEurope=Polygon([(-70, 36), (-10, 36), (-10, 70), (-70, 70)]),
)
basins["Sainsbury2022MWR"] = gpd.GeoDataFrame(index=B.keys(), geometry=list(B.values()))

# Knutson et al. (2020)
# Tropical Cyclones and Climate Change Assessment: Part II: Projected Response to
# Anthropogenic Warming
# https://doi.org/10.1175/BAMS-D-18-0194.1
# Basin definitions from the paper:
# North Atlantic: 0°–90°N, ~265°–360°E (converted: -95°–0°E)
# Northeast Pacific: 0°–90°N, 180°–~265°E (converted: -180°–-95°E)
# Northwest Pacific: 0°–90°N, 100°–180°E
# North Indian: 0°–90°N, 30°–100°E
# South Indian: 90°S–0°, 20°–135°E (converted: -90°–0°, 20°–135°E)
# Southwest Pacific: 90°S–0°, 135°–295°E (converted: -90°–0°, 135°E–-65°E)
# South Atlantic: 90°S–0°, South America to Africa (~-65°–20°E)
# *The North Atlantic–northeast Pacific boundary is on a diagonal tracing a path
# through Mexico and Central America (approximated here similar to WMO convention).
B = dict(
    NATL=Polygon(((-95, 90), (0, 90), (0, 0), (-95, 0))),
    ENP=Polygon(((-180, 0), (-180, 90), (-95, 90), (-95, 0))),
    WNP=Polygon(((100, 0), (100, 90), (180, 90), (180, 0))),
    NI=Polygon(((30, 0), (30, 90), (100, 90), (100, 0))),
    SI=Polygon(((20, -90), (20, 0), (135, 0), (135, -90))),
    SP=MultiPolygon(
        [
            Polygon([(135, 0), (135, -90), (180, -90), (180, 0)]),
            Polygon(((-180, 0), (-180, -90), (-65, -90), (-65, 0))),
        ]
    ),
    SA=Polygon(((-65, -90), (-65, 0), (20, 0), (20, -90))),
)
basins["Knutson2020"] = gpd.GeoDataFrame(index=B.keys(), geometry=list(B.values()))
