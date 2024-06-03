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
from cartopy.crs import Geodetic, PlateCarree

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
def get_basin(lon, lat, convention="WMO", crs=None):
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
    if crs is None:
        crs = Geodetic()
    xyz = PlateCarree().transform_points(crs, lon, lat)

    B = basins_def[convention]  # Select GeoDataFrame for the convention
    points = pd.DataFrame(
        dict(coords=list(zip(xyz[:, 0], xyz[:, 1])))
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
def _get_natural_earth_feature(lon, lat, feature, category, name, resolution, crs=None):
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

    if crs is None:
        crs = Geodetic()
    xyz = PlateCarree().transform_points(crs, lon, lat)
    lon = xyz[:, 0]
    lat = xyz[:, 1]

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


def get_land_or_ocean(lon, lat, resolution="10m", crs=None):
    is_ocean = _get_natural_earth_feature(
        lon,
        lat,
        feature="featurecla",
        category="physical",
        name="ocean",
        resolution=resolution,
        crs=crs,
    )

    is_ocean[is_ocean == ""] = "Land"

    return is_ocean


def get_country(lon, lat, resolution="10m", crs=None):
    return _get_natural_earth_feature(
        lon,
        lat,
        feature="NAME",
        category="cultural",
        name="admin_0_countries",
        resolution=resolution,
        crs=crs,
    )
