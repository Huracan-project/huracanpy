"""
Utils related to geographical attributes
"""

import warnings
from pint.errors import UnitStrippedWarning

import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from cartopy.io.shapereader import natural_earth
from metpy.xarray import preprocess_and_wrap
from cartopy.crs import Geodetic, PlateCarree

from .._basins import basins


@preprocess_and_wrap(wrap_like="lat")
def hemisphere(lat):
    """
    Function to detect which hemisphere each point corresponds to

    Parameters
    ----------
    lat : xarray.DataArray

    Returns
    -------
    xarray.DataArray
        The hemisphere series.
        You can append it to your tracks by running tracks["hemisphere"] = get_hemisphere(tracks)
    """

    return np.where(lat >= 0, "N", "S")


def basin(lon, lat, convention="WMO-TC", crs=None):
    """
    Function to determine the basin of each point, according to the selected convention.

    Parameters
    ----------
    lon : xarray.DataArray
        Longitude series
    lat : xarray.DataArray
        Latitude series
    convention : str
        Name of the basin convention you want to use.
            * WMO
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


@preprocess_and_wrap(wrap_like="lon")
def _get_natural_earth_feature(lon, lat, feature, category, name, resolution, crs=None):
    key = f"{category}_{name}_{resolution}_{feature}"
    if key in _natural_earth_feature_cache:
        df = _natural_earth_feature_cache[key]
    else:
        fname = natural_earth(resolution=resolution, category=category, name=name)
        df = gpd.read_file(fname)
        df = df[["geometry", feature]]
        _natural_earth_feature_cache[key] = df

    # The metpy wrapper converting to pint causes errors, but I'm still going to use it
    # because it lets me pass different array_like types for lon/lat without writing
    # our own wrapper. For now, just convert anything not a numpy array to a numpy array
    if not isinstance(lon, np.ndarray):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UnitStrippedWarning)
            lon = np.array(lon)
            lat = np.array(lat)

    if crs is None:
        crs = Geodetic()
    xyz = PlateCarree().transform_points(crs, lon, lat)

    # Create dataframe of points coordinates
    points = pd.DataFrame(dict(coords=list(xyz[:, :2])))
    # Transform into Points within a GeoDataFrame
    points = gpd.GeoDataFrame(points.coords.apply(Point), geometry="coords", crs=df.crs)

    result = np.array(
        gpd.tools.sjoin(df, points, how="right", predicate="contains")[feature]
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
