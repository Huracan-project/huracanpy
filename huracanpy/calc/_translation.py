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


def _get_distance_azimuth_geod(lon1, lat1, lon2, lat2, ellps="WGS84"):
    # initialize Geod object
    geodesic = pyproj.Geod(ellps=ellps)

    # Compute distance for all data
    fwd_azimuth, back_azimuth, dist = geodesic.inv(lon1, lat1, lon2, lat2)

    return dist, fwd_azimuth


def _get_distance_haversine(lon1, lat1, lon2, lat2):
    # Convert longitudes beyond 180° (necessary for haversine to work)
    lon1 = ((lon1 + 180) % 360) - 180
    lon2 = ((lon2 + 180) % 360) - 180

    yx1 = np.array([lat1, lon1]).T
    yx2 = np.array([lat2, lon2]).T
    return haversine_vector(yx1, yx2, unit="m")


@preprocess_and_wrap(wrap_like="lon")
def azimuth(lon, lat, track_id=None, ellps="WGS84"):
    """Compute azimuth between points using geodesic calculation.

    Parameters
    ----------
    lon : xarray.DataArray
    lat : xarray.DataArray
    track_id : array_like, optional
    ellps : str, optional
        The definition of the globe to use for the geodesic calculation (see
        `pyproj.Geod`). Default is "WGS84".
    Returns
    -------
    xarray.DataArray
        Azimuth in degrees.
            0° corresponds to northward (or stagnating);
            90° corresponds to eastward;
            180° corresponds to southward;
            -90° corresponds to westwards.
    """

    # Compute azimuth
    _, azimuth = _get_distance_azimuth_geod(
        lon[:-1], lat[:-1], lon[1:], lat[1:], ellps=ellps
    )

    # Mask track transition points
    if track_id is not None:
        azimuth[track_id[1:] != track_id[:-1]] = np.nan * azimuth[0]
        azimuth = np.concatenate([azimuth, [np.nan * azimuth[0]]])
    else:
        warnings.warn(
            "track_id is not provided, all points are considered to come from the"
            "same track"
        )

    return azimuth


@dequantify_results
@preprocess_and_wrap(wrap_like="lon")
def distance(lon, lat, *args, track_id=None, method="geod", ellps="WGS84"):
    """Compute distance between longitude/latitude coordinates using
    geodesic or haversine calculation

    >>> distance(lon, lat, track_id)
    Computes the distance between successive lon, lat points, without including
    differences between the end and start points of different tracks

    >>> distance(lon1, lat1, lon2, lat2)
    Computes the distance between each point in (lon1, lat1) and each point in
    (lon2, lat2)

    Parameters
    ----------
    lon : xarray.DataArray
    lat : xarray.DataArray
    *args : xarray.DataArray
        * 0 arguments. Leave empty to calculate distance between successive points
        * 1 argument, track_id. Same as 0 arguments but inserts NaNs where successive
            points are from different tracks
        * 2 arguments, lon and lat arrays. Calculate distances between two tracks,
            e.g. radius of maximum wind speed, using storm centre locations and maximum
            wind speed locations
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
    if len(args) < 2:
        lon1 = lon[:-1]
        lat1 = lat[:-1]
        lon2 = lon[1:]
        lat2 = lat[1:]

        if len(args) == 1:
            track_id = args[0]

        if track_id is None:
            warnings.warn(
                "track_id is not provided, all points are considered to come from the"
                "same track"
            )

    elif len(args) == 2:
        lon1 = lon
        lat1 = lat
        lon2, lat2 = args

    else:
        raise ValueError(
            "Distance either takes 2 arrays (lon/lat) or 4 arrays 2x(lon/lat)"
        )

    if method == "geod":
        dist, _ = _get_distance_azimuth_geod(lon1, lat1, lon2, lat2, ellps=ellps)
    elif method == "haversine":
        dist = _get_distance_haversine(lon1, lat1, lon2, lat2)
    else:
        raise ValueError(
            f"Method {method} for distance calculation not recognised, use one of"
            f"('geod', 'haversince')"
        )

    if len(args) < 2 and track_id is not None:
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
