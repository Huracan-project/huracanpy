"""
Utils related to translation distance and time
"""

import numpy as np
import xarray as xr
from metpy.units import units

import pyproj
from haversine import haversine


def _get_distance_geod(lon, lat, track_id, ellps="WGS84"):
    assert len(lon) == len(lat) == len(track_id)

    # initialize Geod object
    geodesic = pyproj.Geod(ellps=ellps)

    # Compute distance for all data
    dist = []
    for i in range(len(lon) - 1):
        if track_id[i] == track_id[i + 1]:  # If both points belong to the same track
            fwd_azimuth, back_azimuth, distance = geodesic.inv(
                lon[i], lat[i], lon[i + 1], lat[i + 1]
            )
            dist.append(distance)
        else:
            dist.append(np.nan)

    return xr.DataArray(dist, dims=lon.dims) * units("m")


def _get_distance_haversine(lon, lat, track_id):
    # Convert longitudes beyond 180° (necessary for haversine to work)
    lon = xr.DataArray(
        np.where(lon > 180, lon - 360, lon),
        dims=lon.dims,
    )

    dist = []
    for i in range(len(lon) - 1):
        if track_id[i] == track_id[i + 1]:  # If both points belong to the same track
            dx = haversine(
                (float(lat[i].values), float(lon[i].values)),
                (float(lat[i + 1].values), float(lon[i + 1].values)),
                unit="m",
            )  # Displacement in m
            dist.append(dx)
        else:
            dist.append(np.nan)

    return xr.DataArray(dist, dims=lon.dims) * units("m")


def get_distance(lon, lat, track_id=None, method="geod", ellps="WGS84"):
    if track_id is None:
        track_id = [0] * len(lon)
    if method == "geod":
        return _get_distance_geod(lon, lat, track_id, ellps=ellps)
    elif method == "haversine":
        return _get_distance_haversine(
            lon,
            lat,
            track_id,
        )
