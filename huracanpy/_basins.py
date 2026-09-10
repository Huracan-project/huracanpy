from shapely.geometry import MultiPolygon, Polygon


class _LazyBasins:
    """Dict-like wrapper that lazily creates GeoDataFrames from shapely geometry dicts.

    geopandas and cartopy are only imported when a specific basin convention is first
    accessed, keeping ``import huracanpy`` fast.
    """

    def __init__(self, raw_data):
        self._raw = raw_data
        self._cache = {}

    def _create_gdf(self, name):
        import geopandas as gpd
        from cartopy.crs import Geodetic

        raw = self._raw[name]
        return gpd.GeoDataFrame(
            index=raw.keys(), geometry=list(raw.values()), crs=Geodetic()
        )

    def __getitem__(self, key):
        if key not in self._cache:
            if key not in self._raw:
                raise KeyError(key)
            self._cache[key] = self._create_gdf(key)
        return self._cache[key]

    def __contains__(self, key):
        return key in self._raw

    def __iter__(self):
        return iter(self._raw)

    def __len__(self):
        return len(self._raw)

    def keys(self):
        return self._raw.keys()

    def values(self):
        for key in self._raw:
            yield self[key]

    def items(self):
        for key in self._raw:
            yield key, self[key]


# %% Basin geometry data (shapely only — no geopandas/cartopy at module level)

_raw_basins = {}  # dict of {convention: {basin_name: shapely geometry}}


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
_raw_basins["WMO-TC"] = B

ibtracs = dict(
    NI=Polygon(((30, 0), (30, 90), (100, 90), (100, 0))),
    WP=Polygon(((100, 0), (100, 90), (180, 90), (180, 0))),
    EP=Polygon(((-180, 90), (-100, 90), (-100, 20), (-65, 0), (-180, 0))),
    NA=Polygon(((-100, 90), (30, 90), (30, 0), (-65, 0), (-100, 20))),
    SI=Polygon(((20, -90), (20, 0), (135, 0), (135, -90))),
    SP=MultiPolygon(
        [
            Polygon([(135, 0), (135, -90), (180, -90), (180, 0)]),
            Polygon(((-180, 0), (-180, -90), (-65, -90), (-65, 0))),
        ]
    ),
    SA=Polygon(((-65, -90), (-65, 0), (20, 0), (20, -90))),
)
_raw_basins["ibtracs"] = ibtracs

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
_raw_basins["Sainsbury2022JCLI"] = B

# Sainsbury et. al. (2022)
# Why Do Some Post-Tropical Cyclones Impact Europe?
# https://doi.org/10.1175/MWR-D-22-0111.1
B = dict(
    Europe=Polygon([(-10, 36), (30, 36), (30, 70), (-10, 70)]),
    NoEurope=Polygon([(-70, 36), (-10, 36), (-10, 70), (-70, 70)]),
)
_raw_basins["Sainsbury2022MWR"] = B

# Knutson et al. (2020)
# Tropical Cyclones and Climate Change Assessment: Part II: Projected Response to
# Anthropogenic Warming
# https://doi.org/10.1175/BAMS-D-18-0194.1
B = dict(
    NATL=Polygon(((-100, 90), (0, 90), (0, 0), (-65, 0), (-100, 20))),
    ENP=Polygon(((-180, 90), (-100, 90), (-100, 20), (-65, 0), (-180, 0))),
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
_raw_basins["Knutson2020"] = B

basins = _LazyBasins(_raw_basins)
