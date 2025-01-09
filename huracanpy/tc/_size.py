"""
Module containing functions to compute different size metrics
"""

from haversine import haversine
import xarray as xr


def radius_of_maximum_wind(lon_slp, lat_slp, lon_wind_max, lat_wind_max):
    lon_slp = xr.where(lon_slp > 180, lon_slp - 360, lon_slp)
    lon_wind_max = xr.where(lon_wind_max > 180, lon_wind_max - 360, lon_wind_max)

    def f(a, b, c, d):
        return haversine((b, a), (d, c))

    rmw = xr.DataArray(
        [
            f(
                lon_slp.isel(
                    record=i,
                ),
                lat_slp.isel(
                    record=i,
                ),
                lon_wind_max.isel(
                    record=i,
                ),
                lat_wind_max.isel(
                    record=i,
                ),
            )
            for i in range(len(lon_slp.record))
        ],
        dims=lon_slp.dims,
    )

    return rmw
