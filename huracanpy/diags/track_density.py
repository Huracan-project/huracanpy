"""
Module containing function to compute track densities
"""

import numpy as np
import xarray as xr


def simple_global_histogram(lon, lat, bin_size=5, N_seasons=1):
    """
    Function to compute the track density, based on a simple 2D histogram.


    Parameters
    ----------
    lon : xr.DataArray
        longitude series
    lat : xr.DataArray
        latitude series
    bin_size : int or float, optional
        The size in degree of the bins. The default is 5.
    N_seasons : int, optional
        Number of season (will be used to divide the final results, so that is represents points/year). The default is 1.

    Returns
    -------
    xr.DataArray
        Histogram representing number of point per bin per season.

    """

    # compute 2D histogram
    x = np.arange(0, 360 + bin_size, bin_size)
    y = np.arange(-90, 90 + bin_size, bin_size)
    H, X, Y = np.histogram2d(lon, lat, bins=[x, y])
    # Turn into xarray
    da = xr.DataArray(
        H,
        dims=["lon", "lat"],
        coords={"lon": (X[:-1] + X[1:]) / 2, "lat": (Y[:-1] + Y[1:]) / 2},
    )
    # Format
    da = da.where(da > 0).transpose() / N_seasons
    return da
