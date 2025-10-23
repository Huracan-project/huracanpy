"""
Module containing function to compute track densities
"""

import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde

import warnings


def density(
    lon,
    lat,
    method="histogram",
    bin_size=5,
    lon_range=None,
    lat_range=(-90, 90),
    crop=False,
    function_kws=dict(),
):
    """Function to compute the track density, based on a simple 2D histogram.

    Parameters
    ----------
    lon : array_like
        longitude series
    lat : array_like
        latitude series
    method : str, default="histogram"
        The method used to calculate the density, currently "histogram" or "kde", which
        gives a 2d histogram using `np.histogram2d`
    bin_size : int or float, default=5
        When using histogram, defines the size (in degrees) of the bins.
    lon_range : tuple, default
        The maximum and minimum longitude to calculate the density over. If None, then
        it is set to global: (-180, 180) or (0, 360) depending on the input data
    lat_range : tuple, default=(-90, 90)
        The maximum and minimum latitude to calculate the density over.
    crop : bool, default=False
        If True crop the result to remove any outer bounds that only have zero density
    function_kws : dict
        Keyword arguments passed to the function used for calculating density

        * If method="histogram", `numpy.histogram2d`
        * If method="kde", `scipy.stats.gaussian_kde`

    Raises
    ------
    NotImplementedError
        If method given is not 'histogram' or 'kde'

    Returns
    -------
    xarray.DataArray
        Track density as a 2D map.

    """
    # Define coordinates for mapping
    if lon_range is None:
        if lon.min() < 0:
            lon_range = (-180, 180)
        else:
            lon_range = (0, 360)

    x_edge = np.arange(lon_range[0], lon_range[1] + bin_size, bin_size)
    y_edge = np.arange(lat_range[0], lat_range[1] + bin_size, bin_size)
    x_mid, y_mid = (x_edge[1:] + x_edge[:-1]) / 2, (y_edge[1:] + y_edge[:-1]) / 2

    # Compute density
    if method == "histogram":
        h = _histogram(lon, lat, x_edge, y_edge, function_kws)
    elif method == "kde":
        h = _kde(lon, lat, x_mid, y_mid, function_kws)
    else:
        raise NotImplementedError(
            f"Method {method} not implemented yet. Use one 'histogram', 'kde'"
        )

    # Turn into xarray
    da = xr.DataArray(
        h,
        dims=["lat", "lon"],
        coords={"lon": x_mid, "lat": y_mid},
    )

    if crop:
        # Crop the map to where there are non-zero points
        has_data = da > 0

        # Keep the band of latitudes between first and lat non-empty row
        # and longitudes between first and lat empty column
        idx_lat = np.where(has_data.any(dim="lon"))[0]
        da = da.isel(lat=slice(idx_lat[0], idx_lat[-1] + 1))

        idx_lon = np.where(has_data.any(dim="lat"))[0]
        da = da.isel(lon=slice(idx_lon[0], idx_lon[-1] + 1))

        return da
    else:
        return da


def _histogram(lon, lat, x_edge, y_edge, function_kws):
    # Compute 2D histogram with numpy
    h, _x, _y = np.histogram2d(lon, lat, bins=[x_edge, y_edge], **function_kws)
    return h.T  # Transpose result


def _kde(lon, lat, x_mid, y_mid, function_kws):
    # engineer positions array for kernel estimation computation
    positions = np.reshape(np.meshgrid(x_mid, y_mid), (2, len(x_mid) * len(y_mid)))
    # Compute kernel density estimate
    kernel = gaussian_kde([lon, lat], **function_kws)
    # Evaluation kernel along positions
    h = np.reshape(kernel(positions), (len(y_mid), len(x_mid)))
    # Account for cell area differences
    warnings.warn(
        "The kde function does not currently take into account the spherical "
        "geometry of the Earth."
    )
    # Normalize so that H integrates to the total number of points
    return h * len(lon) / h.sum()
