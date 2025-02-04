"""
Module containing function to compute track densities
"""

import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde


def density(lon, lat, method="histogram", bin_size=5, crop=False, function_kws=dict()):
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

    Raises
    ------
    NotImplementedError
        If method given is not 'histogram'

    Returns
    -------
    xarray.DataArray
        Track density as a 2D map.

    """
    # Define coordinates for mapping
    if lon.min() < 0:
        lon_range = (-180, 180)
    else:
        lon_range = (0, 360)
    x_edge = np.arange(lon_range[0], lon_range[1] + bin_size, bin_size)
    y_edge = np.arange(-90, 90 + bin_size, bin_size)
    x_mid, y_mid = (x_edge[1:] + x_edge[:-1]) / 2, (y_edge[1:] + y_edge[:-1]) / 2

    # Compute density
    if method == "histogram":
        H = _histogram(lon, lat, x_edge, y_edge, function_kws)
    elif method == "kde":
        H = _kde(lon, lat, x_mid, y_mid, function_kws)
    else:
        raise NotImplementedError(
            f"Method {method} not implemented yet. Use one 'histogram'"
        )

    # Turn into xarray
    da = xr.DataArray(
        H,
        dims=["lat", "lon"],
        coords={"lon": x_mid, "lat": y_mid},
    )

    if crop:  # Crop the map to where there are non-zero points
        da = da.where(da > 0)
        return da.where(~np.isnan(da), drop=True).fillna(0)
    else:
        return da


def _histogram(lon, lat, x_edge, y_edge, function_kws):
    # Compute 2D histogram with numpy
    H, _x, _y = np.histogram2d(lon, lat, bins=[x_edge, y_edge], **function_kws)
    return H.T  # Transpose result


def _kde(lon, lat, x_mid, y_mid, function_kws):
    # engineer positions array for kernel estimation computation
    positions = np.reshape(np.meshgrid(x_mid, y_mid), (2, len(x_mid) * len(y_mid)))
    # Compute kernel density estimate
    kernel = gaussian_kde([lon, lat], **function_kws)
    # Evaluation kernel along positions
    return np.reshape(kernel(positions), (len(y_mid), len(x_mid)))
