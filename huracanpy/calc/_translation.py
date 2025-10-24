"""
Utils related to translation distance and time
"""

import warnings

from haversine import haversine_vector
from metpy.units import units
from metpy.xarray import preprocess_and_wrap
import numpy as np
from pint.errors import UnitStrippedWarning
import pyproj


from ._rates import delta, _dummy_track_id, _align_array
from .._metpy import dequantify_results


def _get_distance_azimuth_geod(lon1, lat1, lon2, lat2, ellps="WGS84"):
    # initialize Geod object
    geodesic = pyproj.Geod(ellps=ellps)

    # Compute distance for all data
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UnitStrippedWarning,
            message="The unit of the quantity is stripped when downcasting to ndarray.",
        )
        fwd_azimuth, back_azimuth, dist = geodesic.inv(lon1, lat1, lon2, lat2)

    return dist * units("m"), fwd_azimuth * units("degrees")


def _get_distance_haversine(lon1, lat1, lon2, lat2):
    # Convert longitudes beyond 180° (necessary for haversine to work)
    lon1 = ((lon1 + 180) % 360) - 180
    lon2 = ((lon2 + 180) % 360) - 180

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UnitStrippedWarning,
            message="The unit of the quantity is stripped when downcasting to ndarray.",
        )
        yx1 = np.asarray([lat1, lon1]).T
        yx2 = np.asarray([lat2, lon2]).T

    return haversine_vector(yx1, yx2, unit="m") * units("m")


@dequantify_results
@preprocess_and_wrap(wrap_like="lon")
def azimuth(lon, lat, track_id=None, ellps="WGS84", centering="forward"):
    """Compute azimuth between points using geodesic calculation.

    Parameters
    ----------
    lon : xarray.DataArray
    lat : xarray.DataArray
    track_id : array_like, optional
    ellps : str, optional
        The definition of the globe to use for the geodesic calculation (see
        `pyproj.Geod`). Default is "WGS84".
    centering: str, optional
        - "forward" gives the angle based on the track point and the following track
          point. The last point of each track will be NaN
        - "backward" gives the angle based on the track point and the previous track
          point. The first point of each track will be NaN
        - "centre" gives the angle based on the centred difference of track points. The
          first and last points of each track will be NaN
        - "adaptive" gives the same as centred, but fills in the first point of each
          track with the forward difference, and the last point of each track with the
          backward difference

    Returns
    -------
    xarray.DataArray
        Azimuth in degrees.
            0° corresponds to northward (or stagnating);
            90° corresponds to eastward;
            180° corresponds to southward;
            -90° corresponds to westwards.
    """
    if track_id is None:
        track_id = _dummy_track_id(lon)

    # Compute azimuth
    _, fwd_azimuth = _get_distance_azimuth_geod(
        lon[:-1], lat[:-1], lon[1:], lat[1:], ellps=ellps
    )

    if centering in ["forward", "backward"]:
        return _align_array(fwd_azimuth, track_id, centering)

    else:
        # Compute angle in steps of two
        _, centred_azimuth = _get_distance_azimuth_geod(
            lon[:-2], lat[:-2], lon[2:], lat[2:], ellps=ellps
        )
        centred_azimuth[track_id[2:] != track_id[:-2]] = np.nan * centred_azimuth[0]

        return _align_array(fwd_azimuth, track_id, centering, centred_azimuth)


@dequantify_results
@preprocess_and_wrap(wrap_like="lon")
def distance(
    lon, lat, *args, track_id=None, method="geod", ellps="WGS84", centering="forward"
):
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
        - 0 arguments. Leave empty to calculate distance between successive points
        - 1 argument, track_id. Same as 0 arguments but inserts NaNs where successive
          points are from different tracks
        - 2 arguments, lon and lat arrays. Calculate distances between two tracks,
          e.g. radius of maximum wind speed, using storm centre locations and maximum
          wind speed locations
    track_id : array_like, optional
    method : str, optional
        The method of computing distances, either geodesic (`"geod"`/`"geodesic"`) or
        haversine (`"haversine"`)
    ellps : str, optional
        The definition of the globe to use for the geodesic calculation (see
        `pyproj.Geod`). Default is "WGS84".
    centering: str, optional
        - "forward" gives the distance based on the track point and the following track
          point. The last point of each track will be NaN
        - "backward" gives the distance based on the track point and the previous track
          point. The first point of each track will be NaN
        - "centre" gives the distance based on the centred difference of track points.
          The first and last points of each track will be NaN
        - "adaptive" gives the same as centred, but fills in the first point of each
          track with the forward difference, and the last point of each track with the
          backward difference

    Returns
    -------
    xarray.DataArray

    """
    # Curate input
    # If track_id is not provided, all points are considered to belong to the same track
    if len(args) < 2:
        lon1 = lon[:-1]
        lat1 = lat[:-1]
        lon2 = lon[1:]
        lat2 = lat[1:]

        if len(args) == 1:
            if track_id is None:
                track_id = args[0]
            else:
                raise ValueError(
                    "Distance either takes 2 arrays (lon/lat) or 4 arrays 2x(lon/lat)"
                )

        if track_id is None:
            track_id = _dummy_track_id(lon)

    elif len(args) == 2:
        lon1 = lon
        lat1 = lat
        lon2, lat2 = args

    else:
        raise ValueError(
            "Distance either takes 2 arrays (lon/lat) or 4 arrays 2x(lon/lat)"
        )

    if method in ["geod", "geodesic"]:
        dist, _ = _get_distance_azimuth_geod(lon1, lat1, lon2, lat2, ellps=ellps)
    elif method == "haversine":
        dist = _get_distance_haversine(lon1, lat1, lon2, lat2)
    else:
        raise ValueError(
            f"Method {method} for distance calculation not recognised, use one of"
            f"('geod', 'haversine')"
        )

    if len(args) < 2 and track_id is not None:
        dist = _align_array(dist, track_id, centering)

    return dist


@dequantify_results
@preprocess_and_wrap(wrap_like="lon")
def translation_speed(
    lon, lat, time, track_id=None, method="geod", ellps="WGS84", centering="forward"
):
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
    centering: str, optional
        - "forward" gives the speed based on the track point and the following track
          point. The last point of each track will be NaN
        - "backward" gives the speed based on the track point and the previous track
          point. The first point of each track will be NaN
        - "centre" gives the speed based on the centred difference of track points. The
          first and last points of each track will be NaN
        - "adaptive" gives the same as centred, but fills in the first point of each
          track with the forward difference, and the last point of each track with the
          backward difference

    Returns
    -------
    xarray.DataArray

    """
    # Curate input
    # If track_id is not provided, all points are considered to belong to the same track
    if track_id is None:
        track_id = _dummy_track_id(lon)

    # Distance between each points
    dx = distance(
        lon, lat, track_id=track_id, method=method, ellps=ellps, centering=centering
    )

    # time between each points
    dt = delta(time, track_id, centering=centering)

    return dx / dt
