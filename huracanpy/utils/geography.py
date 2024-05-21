"""
Utils related to geographical attributes
"""

import numpy as np
import xarray as xr
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

from ._basins import basins_def


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

    H = np.where(lat >= 0, "N", "S")
    return xr.DataArray(H, dims="obs", coords={"obs": lat.obs})


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
        dict(coords=list(zip(lon.values, lat.values)))
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
    return xr.DataArray(basin, dims="obs", coords={"obs": lon.obs})
