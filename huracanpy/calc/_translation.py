"""
Utils related to translation distance and time
"""

import warnings

import numpy as np
from metpy.units import units
from metpy.xarray import preprocess_and_wrap

import pyproj
from haversine import haversine_vector

from ._rates import delta
from .._metpy import dequantify_results


def _get_distance_geod(lon, lat, ellps="WGS84"):
    # initialize Geod object
    geodesic = pyproj.Geod(ellps=ellps)

    # Compute distance for all data
    fwd_azimuth, back_azimuth, dist = geodesic.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])

    return dist


def _get_distance_haversine(lon, lat):
    # Convert longitudes beyond 180° (necessary for haversine to work)
    lon = ((lon + 180) % 360) - 180

    yx = np.array([lat, lon]).T
    return haversine_vector(yx[:-1], yx[1:], unit="m")


@dequantify_results
@preprocess_and_wrap(wrap_like="lon")
def distance(lon, lat, track_id=None, method="geod", ellps="WGS84"):
    """Compute the distance between successive lon, lat points, without including
    differences between the end and start points of different tracks

    Parameters
    ----------
    lon : xarray.DataArray
    lat : xarray.DataArray
    track_id : array_like, optional
    method : str, optional
        The method of computing distances, either geodesic (`"geod"`) or haversine
        (`"haversine"`)
    ellps : str, optional
        The definition of the globe to use for the geodesic calculation (see
        `pyproj.Geod`). Default is "WGS84".

    Returns
    -------
    xarray.DataArray

    """
    # TODO: Provide option for centering forward, backwards, centered

    # Curate input
    # If track_id is not provided, all points are considered to belong to the same track
    if track_id is None:
        track_id = [0] * len(lon)
        warnings.warn(
            "track_id is not provided, all points are considered to come from the same track"
        )

    if method == "geod":
        dist = _get_distance_geod(lon, lat, ellps=ellps)
    elif method == "haversine":
        dist = _get_distance_haversine(lon, lat)
    else:
        raise ValueError(
            f"Method {method} for distance calculation not recognised, use one of"
            f"('geod', 'haversince')"
        )

    dist[track_id[1:] != track_id[:-1]] = np.nan * dist[0]
    dist = np.concatenate([dist, [np.nan * dist[0]]])

    return dist * units("m")


@dequantify_results
@preprocess_and_wrap(wrap_like="lon")
def translation_speed(lon, lat, time, track_id=None, method="geod", ellps="WGS84"):
    """
    Compute translation speed along tracks

    Parameters
    ----------
    lon : xarray.DataArray
    lat : xarray.DataArray
    time : xarray.DataArray
    track_id : array_like, optional
    method : str, optional
        The method of computing distances, either geodesic (`"geod"`) or haversine
        (`"haversine"`)
    ellps : str, optional
        The definition of the globe to use for the geodesic calculation (see
        `pyproj.Geod`). Default is "WGS84".


    Returns
    -------
    xarray.DataArray

    """
    # TODO: Provide option for centering forward, backwards, centered

    # Curate input
    # If track_id is not provided, all points are considered to belong to the same track
    if track_id is None:
        np.zeros_like(lon)
        warnings.warn(
            "track_id is not provided, all points are considered to come from the same"
            "track"
        )

    # Distance between each points
    dx = distance(lon, lat, track_id, method=method, ellps=ellps)

    # time between each points
    dt = delta(time, track_id)

    return dx / dt
