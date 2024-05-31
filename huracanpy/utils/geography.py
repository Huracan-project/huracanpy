"""
Utils related to geographical attributes
"""

import warnings
from pint.errors import UnitStrippedWarning

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
import geopandas as gpd
from cartopy.io.shapereader import natural_earth
from metpy.xarray import preprocess_and_wrap

from ._basins import basins_def


@preprocess_and_wrap(wrap_like="lat")
def get_hemisphere(lat):
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


@preprocess_and_wrap(wrap_like="lon")
def get_basin(lon, lat, convention="WMO"):
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

    Returns
    -------
    xarray.DataArray
        The basin series.
        You can append it to your tracks by running tracks["basin"] = get_basin(tracks)
    """

    B = basins_def[convention]  # Select GeoDataFrame for the convention
    points = pd.DataFrame(
        dict(coords=list(zip(lon, lat)))
    )  # Create dataframe of points coordinates
    points = gpd.GeoDataFrame(
        points.coords.apply(Point), geometry="coords", crs=B.crs
    )  # Transform into Points within a GeoDataFrame
    basin = (
        gpd.tools.sjoin(
            points,
            B,
            how="left",  # Identify basins
        )
        .reset_index()
        .groupby("index")
        .first(  # Deal with points at borders
        )
        .index_right
    )  # Select basin names
    return basin


@preprocess_and_wrap(wrap_like="lon")
def _get_natural_earth_feature(lon, lat, feature, category, name, resolution):
    fname = natural_earth(resolution=resolution, category=category, name=name)
    df = gpd.read_file(fname)

    # The metpy wrapper converting to pint causes errors, but I'm still going to use it
    # because it lets me pass different array_like types for lon/lat without writing
    # our own wrapper. For now, just convert anything not a numpy array to a numpy array
    if not isinstance(lon, np.ndarray):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UnitStrippedWarning)
            lon = np.array(lon)
            lat = np.array(lat)

    # Any strings are loaded in as objects. Use the specific string type with the
    # maximum possible length for the output instead
    dtype = df[feature].dtype
    if dtype == "O":
        max_length = df[feature].apply(len).max()
        dtype = f"U{max_length}"

    result = np.zeros(len(lon), dtype=dtype)
    for n, row in df.iterrows():
        result[np.where(shapely.contains_xy(row.geometry, lon, lat))] = row[feature]

    return result


def get_land_or_ocean(lon, lat, resolution="10m"):
    is_ocean = _get_natural_earth_feature(
        lon,
        lat,
        feature="featurecla",
        category="physical",
        name="ocean",
        resolution=resolution,
    )

    is_ocean[is_ocean == ""] = "Land"

    return is_ocean


def get_country(lon, lat, resolution="10m"):
    return _get_natural_earth_feature(
        lon,
        lat,
        feature="NAME",
        category="cultural",
        name="admin_0_countries",
        resolution=resolution,
    )
