"""
Utils related to geographical attributes
"""

import warnings

from cartopy.io.shapereader import natural_earth
import geopandas as gpd
from metpy.xarray import preprocess_and_wrap
import numpy as np
from pint.errors import UnitStrippedWarning
import xarray as xr

from .._basins import basins
from ..convert import to_geodataframe


@preprocess_and_wrap(wrap_like="lat")
def hemisphere(lat):
    """
    Function to detect which hemisphere each point corresponds to.

    Parameters
    ----------
    lat : xarray.DataArray

    Returns
    -------
    xarray.DataArray
        The hemisphere series.
        You can append it to your tracks by running

        >>> tracks["hemisphere"] = get_hemisphere(tracks.lat)
    """

    return np.where(lat >= 0, "N", "S")


def basin(lon, lat, convention="WMO-TC", crs=None):
    """
    Function to determine the basin of each point, according to the selected convention.

    Parameters
    ----------
    lon : float or array_like
        Longitude series
    lat : float or array_like
        Latitude series
    convention : str
        Name of the basin convention you want to use.
            * **WMO-TC** - WMO defined tropical cyclone basins
            * **Sainsbury2022JCLI** - Definitions from
                (https://doi.org/10.1175/JCLI-D-21-0712.1) North Atlantic split up into:

                * Main development region (MDR)
                * Subtropical development region (SUB)
                * Western basin / Caribbean sea (WEST)

            * **Sainsbury2022MWR** - Definitions from
                (https://doi.org/10.1175/MWR-D-22-0111.1). Extratropical transition in
                North Atlantic divided into:

                * Europe
                * NoEurope

            * **Knutson2020** - Definitions from
                (https://doi.org/10.1175/BAMS-D-18-0194.1). Global basins:

                * NATL (North Atlantic)
                * ENP (Northeast Pacific)
                * WNP (Northwest Pacific)
                * NI (North Indian)
                * SI (South Indian)
                * SP (Southwest Pacific)
                * SA (South Atlantic)

    crs : cartopy.crs.CRS, optional
        The coordinate reference system of the lon, lat inputs. The basins are defined
        in PlateCarree (-180, 180), so this will transform lon/lat to this projection
        before checking the basin. If None is given, it will use cartopy.crs.Geodetic
        which is essentially the same, but allows the longitudes to be defined in ranges
        broader than -180, 180

    Returns
    -------
    xarray.DataArray
        The basin series.
        You can append it to your tracks by running tracks["basin"] = get_basin(tracks)
    """
    return _get_natural_earth_feature(
        lon,
        lat,
        feature="basin",
        category="physical",
        name=convention,
        resolution=0,
        crs=crs,
    )


# Running this on lots of tracks was very slow if the file is reopened every time this
# is called
_natural_earth_feature_cache = {
    f"physical_{key}_0_basin": value.rename_axis("basin").reset_index()
    for key, value in basins.items()
}


def _cache_natural_earth_feature(feature, category, name, resolution):
    key = f"{category}_{name}_{resolution}_{feature}"
    if key in _natural_earth_feature_cache:
        df = _natural_earth_feature_cache[key]
    else:
        fname = natural_earth(resolution=resolution, category=category, name=name)
        df = gpd.read_file(fname)
        df = df[["geometry", feature]]
        _natural_earth_feature_cache[key] = df

    return df


@preprocess_and_wrap(wrap_like="lon")
def _get_natural_earth_feature(
    lon,
    lat,
    feature,
    category,
    name,
    resolution,
    predicate="contains",
    track_id=None,
    crs=None,
):
    # The metpy wrapper converting to pint causes errors, but I'm still going to use it
    # because it lets me pass different array_like types for lon/lat without writing
    # our own wrapper. For now, just convert anything not a numpy array to a numpy array
    if not isinstance(lon, np.ndarray):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UnitStrippedWarning)
            lon = np.asarray(lon)
            lat = np.asarray(lat)

            if track_id is not None:
                track_id = np.asarray(track_id)

    df = _cache_natural_earth_feature(feature, category, name, resolution)

    tracks = to_geodataframe(lon, lat, track_id, crs=crs).to_crs(df.crs)

    result = np.asarray(
        gpd.tools.sjoin(df, tracks, how="right", predicate=predicate)[feature]
    ).astype(str)

    # Set "nan" as empty
    result[result == "nan"] = ""

    return result


def is_ocean(lon, lat, resolution="10m", crs=None):
    """
    Detect whether each point is over ocean

    Parameters
    ----------
    lon, lat : float or array_like
    resolution : str
        The resolution of the Land/Sea outlines dataset to use. One of

        * 10m (1:10,000,000)
        * 50m (1:50,000,000)
        * 110m (1:110,000,000)

    crs : cartopy.crs.CRS

    Returns
    -------
    array_like[bool]
        Array of "Land" or "Ocean" for each lon/lat point. Should return the same type
        of array as the input lon/lat, or a length 1 :py:class:`numpy.ndarray` if
        lon/lat are floats
    """
    return (
        _get_natural_earth_feature(
            lon,
            lat,
            feature="featurecla",
            category="physical",
            name="ocean",
            resolution=resolution,
            crs=crs,
        )
        == "Ocean"
    )


def is_land(lon, lat, resolution="10m", crs=None):
    """
    Detect whether each point is over land

    Parameters
    ----------
    lon, lat : float or array_like
    resolution : str
        The resolution of the Land/Sea outlines dataset to use. One of

        * 10m (1:10,000,000)
        * 50m (1:50,000,000)
        * 110m (1:110,000,000)

    crs : cartopy.crs.CRS

    Returns
    -------
    array_like[bool]
        Array of "Land" or "Ocean" for each lon/lat point. Should return the same type
        of array as the input lon/lat, or a length 1 :py:class:`numpy.ndarray` if
        lon/lat are floats
    """
    return (
        _get_natural_earth_feature(
            lon,
            lat,
            feature="featurecla",
            category="physical",
            name="ocean",
            resolution=resolution,
            crs=crs,
        )
        == ""
    )


def country(lon, lat, resolution="10m", crs=None):
    """Detect the country each point is over

    Parameters
    ----------
    lon, lat : float or array_like
    resolution : str
        The resolution of the Land/Sea outlines dataset to use. One of

        * 10m (1:10,000,000)
        * 50m (1:50,000,000)
        * 110m (1:110,000,000)

    crs : cartopy.crs.CRS

    Returns
    -------
    array_like
        Array of country names (or empty string for no country) for each lon/lat point.
        Should return the same type of array as the input lon/lat, or a length 1
        :py:class:`numpy.ndarray` if lon/lat are floats
    """
    return _get_natural_earth_feature(
        lon,
        lat,
        feature="NAME",
        category="cultural",
        name="admin_0_countries",
        resolution=resolution,
        crs=crs,
    )


def continent(lon, lat, resolution="10m", crs=None):
    """Detect the continent each point is over

    Parameters
    ----------
    lon, lat : float or array_like
    resolution : str
        The resolution of the Land/Sea outlines dataset to use. One of

        * 10m (1:10,000,000)
        * 50m (1:50,000,000)
        * 110m (1:110,000,000)

    crs : cartopy.crs.CRS

    Returns
    -------
    array_like
        Array of continent names (or empty string for no continent) for each lon/lat
        point. Should return the same type of array as the input lon/lat, or a length 1
        :py:class:`numpy.ndarray` if lon/lat are floats
    """
    return _get_natural_earth_feature(
        lon,
        lat,
        feature="CONTINENT",
        category="cultural",
        name="admin_0_countries",
        resolution=resolution,
        crs=crs,
    )


@preprocess_and_wrap()
def landfall_points(lon, lat, track_id=None, *, resolution="10m", crs=None):
    """Find the points where the tracks intersect with a coastline

    Parameters
    ----------
    lon, lat : float or array_like
    track_id : float or array_like, optional
    resolution : str
        The resolution of the Land/Sea outlines dataset to use. One of

        * 10m (1:10,000,000)
        * 50m (1:50,000,000)
        * 110m (1:110,000,000)

        Default is "10m"
    crs : cartopy.crs.CRS
        The coordinate system that the input lon/lat points are in. Default is None,
        which assumes Geodesic with Earth radius.

    Returns
    -------
    xarray.Dataset

    """
    df = _cache_natural_earth_feature("featurecla", "physical", "coastline", resolution)

    tracks = to_geodataframe(lon, lat, track_id, crs=crs).to_crs(df.crs)

    # Get the combinations of track_id / coastline that have intersections
    result = gpd.tools.sjoin(tracks, df, predicate="intersects")

    # For each combination of track_id / coastline get the exact point(s) that they
    # intersect and save as a set of tracks in the same record, track_id format
    points = []
    for n, row in result.iterrows():
        track_id = tracks.loc[n].track_id
        track = gpd.GeoSeries(tracks.loc[n].geometry, crs=tracks.crs)
        coastline = gpd.GeoSeries(df.loc[row.index_right].geometry, crs=df.crs)

        points += [
            (track_id, p.x, p.y) for p in track.intersection(coastline).explode()
        ]

    points = np.asarray(points)

    return xr.Dataset(
        data_vars=dict(
            track_id=("record", points[:, 0]),
            lon=("record", points[:, 1]),
            lat=("record", points[:, 2]),
        )
    )
