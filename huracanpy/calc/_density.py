"""
Module containing function to compute track densities
"""

import numpy as np
import xarray as xr


def density(lon, lat, *, method="histogram", bin_size=5, n_seasons=1):
    """Function to compute the track density, based on a simple 2D histogram.

    Parameters
    ----------
    lon : array_like
        longitude series
    lat : array_like
        latitude series
    method : str, default="histogram"
        The method used to calculate the density, currently only "histogram", which
        gives a 2d histogram using `np.histogram2d`
    bin_size : int or float, default=5
        When using histogram, defines the size (in degrees) of the bins.
    n_seasons : int, optional
        Number of season (will be used to divide the final results, so that is
        represents points/year). The default is 1.

    Returns
    -------
    xarray.DataArray
        Histogram representing number of point per bin per season.

    """
    if method == "histogram":
        return _histogram(lon, lat, bin_size=bin_size) / n_seasons
    else:
        raise NotImplementedError(
            f"Method {method} not implemented yet. Use one 'histogram'"
        )


def _histogram(lon, lat, bin_size=5):
    if lon.min() < 0:
        lon_range = (-180, 180)
    else:
        lon_range = (0, 360)
    # compute 2D histogram
    x = np.arange(lon_range[0], lon_range[1] + bin_size, bin_size)
    y = np.arange(-90, 90 + bin_size, bin_size)
    H, X, Y = np.histogram2d(lon, lat, bins=[x, y])
    # Turn into xarray
    da = xr.DataArray(
        H,
        dims=["lon", "lat"],
        coords={"lon": (X[:-1] + X[1:]) / 2, "lat": (Y[:-1] + Y[1:]) / 2},
    )
    # Format
    da = da.where(da > 0).transpose()
    return da.where(~np.isnan(da), drop=True).fillna(0)
