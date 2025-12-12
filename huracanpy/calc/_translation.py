"""
Utils related to translation distance and time
"""

from datetime import timedelta
import shapely
import warnings

from cartopy.crs import Geodetic, Orthographic
from haversine import haversine_vector
from metpy.units import units
from metpy.xarray import preprocess_and_wrap
import nvector
import numpy as np
import pandas as pd
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
    lon, lat : xarray.DataArray
        Longitude and latitude points
    track_id : array_like, optional
        Track ID at each point
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
    lon, lat : xarray.DataArray
        Longitude and latitude points
    *args : xarray.DataArray
        - 0 arguments. Leave empty to calculate distance between successive points
        - 1 argument, track_id. Same as 0 arguments but inserts NaNs where successive
          points are from different tracks
        - 2 arguments, lon and lat arrays. Calculate distances between two tracks,
          e.g. radius of maximum wind speed, using storm centre locations and maximum
          wind speed locations
    track_id : array_like, optional
        Track ID at each point
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
    lon, lat : xarray.DataArray
        Longitude and latitude points
    time : xarray.DataArray
        Time for each point
    track_id : array_like, optional
        Track ID at each points
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


@dequantify_results
@preprocess_and_wrap(wrap_like="lon")
def corral_radius(lon, lat, time=None, track_id=None, *, window=None, min_points=None):
    """Find the minimum radius encircling a set of points

    By default, calling corral radius with a set of lons/lats will return the corral
    radius for all these points. The returned array with have the same length as
    lons/lats, but the same value for all points

    >>> corral_radius(lons, lats)

    If you also pass a track_id, the corral radius is calculated separately for each
    unique track. The returned array still has the same length as lons/lats so the value
    for each track is repeated

    >>> corral_radius(lons, lats, track_id=track_id)

    Passing a time and window will calculate the corral radius in a rolling window. The
    window is by default in hours (you can explicitly pass a datetime.timedelta to use
    a more specific window). The code below will calculate the corral radius for each
    lon/lat include the lons/lats withing +/- 36 hours. Points where the window is
    outside the times are given NaNs

    >>> corral_radius(lons, lats, time=time, window=36)

    Including both a track_id and a time/window will ensure the corral radius is
    calculated separately for each track, leaving NaNs at the start and end of each
    track where the time window is outside the track times

    >>> corral_radius(lons, lats, time=time, track_id=track_id, window=36)


    Parameters
    ----------
    lon, lat : array_like
        Longitude and latitude points
    time : array_like, optional
        Time at each point
    track_id : array_like, optional
        Track ID at each point
    window : scalar or datetime.timedelta, optional
        Half-width of the window. i.e. include all times within +/- window
    min_points : int, optional
        Minimum number of points required in the window.

    Returns
    -------
    np.ndarray
        Corral radii in metres (NaN if not computed)
    """
    if track_id is None:
        track_id = _dummy_track_id(lon)

    # Convert time to pandas
    # Comparison below fails when using xarray. It seems to interpret a numpy timedelta
    # of zero as an integer and throws a TypeError
    if time is not None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UnitStrippedWarning)
            time = pd.to_datetime(time)

    if np.isscalar(window):
        window = timedelta(hours=window)

    corral_radii = np.full(len(lon), np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UnitStrippedWarning)
        track_id, indices, counts = np.unique(
            np.asarray(track_id), return_index=True, return_counts=True
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UnitStrippedWarning)
        for idx, count in zip(indices, counts):
            if time is not None and window is not None:
                times = time[idx : idx + count]

                for n, centre_time in enumerate(times):
                    if (centre_time - times[0] >= window) and (
                        times[-1] - centre_time >= window
                    ):
                        subset = np.where(np.abs(times - centre_time) <= window)[0]
                        if min_points is None or len(subset) >= min_points:
                            radius = _make_circle(
                                np.asarray(lon[idx + subset]),
                                np.asarray(lat[idx + subset]),
                            )
                            corral_radii[idx + n] = radius
                    # else: leave as np.nan
            else:
                corral_radii[idx : idx + count] = _make_circle(
                    np.asarray(lon[idx : idx + count]),
                    np.asarray(lat[idx : idx + count]),
                )

    return corral_radii * units("m")


def _make_circle(lons, lats):
    xyz = _latlon_to_xy(lons, lats)
    line = shapely.LineString(xyz[:, :2])

    return shapely.minimum_bounding_radius(line)


def _latlon_to_xy(lon, lat):
    # Convert lat/lon to x/y using Orthographic projection for small areas
    # For more accuracy, use spherical geometry, but for <2000km, this is fine
    # Center for projection
    lon0, lat0 = _centre_point(lon, lat)
    projection = Orthographic(central_longitude=lon0, central_latitude=lat0)

    return projection.transform_points(Geodetic(), lon, lat)


def _centre_point(lons, lats):
    points = nvector.GeoPoint.from_degrees(longitude=lons, latitude=lats)
    centre = points.to_nvector().mean().to_geo_point()
    return centre.longitude_deg, centre.latitude_deg
