"""
Module containing function to compute track densities
"""

import warnings

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from cartopy.crs import Geodetic
from metpy.constants import earth_avg_radius
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

from .._metpy import dequantify_results
from ..convert import to_geodataframe


def _area(x_edge, y_edge):
    return (earth_avg_radius**2) * np.outer(
        np.diff(np.sin(np.deg2rad(y_edge))), np.diff(np.deg2rad(x_edge))
    )


@dequantify_results
def density(
    lon,
    lat,
    track_id=None,
    *,
    method="histogram",
    bin_size=5,
    lon_range=None,
    lat_range=(-90, 90),
    crop=False,
    spherical=False,
    function_kws=None,
    **kwargs,
):
    """Function to compute the track density, based on a simple 2D histogram.

    Parameters
    ----------
    lon : array_like
        longitude series
    lat : array_like
        latitude series
    track_id : array_like:
        Track ID of each if density is calculated by track line intersections with a
        grid (method="line")
    method : str, default="histogram"
        The method used to calculate the density
        - **histogram** gives a 2d histogram using :func:`np.histogram2d`
        - **kde** gives a kernel density estimate using
          :class:`sklearn.neighbors.KernelDensity` if `spherical=True` or
          :obj:`scipy.stats.gaussian_kde` if `spherical=False`
        - **line** gives a 2d histogram where the counts are based if the track crosses
          each gridbox. This means tracks with multiple points in the same gridbox are
          only counted once per gridbox.
    bin_size : int or float, default=5
        When using histogram, defines the size (in degrees) of the bins.
    lon_range : tuple, default
        The maximum and minimum longitude to calculate the density over. If None, then
        it is set to global: (-180, 180) or (0, 360) depending on the input data
    lat_range : tuple, default=(-90, 90)
        The maximum and minimum latitude to calculate the density over.
    crop : bool, default=False
        If True crop the result to remove any outer bounds that only have zero density
    spherical : bool, default=False
        Account for the spherical Earth
    function_kws : dict
        Keyword arguments passed to the function used for calculating density

        * If method="histogram", :func:`numpy.histogram2d`
        * If method="kde" and spherical=`True`,
          :class:`sklearn.neighbors.KernelDensity`. Note that the bandwidth argument is
          set to `"scott"` rather than the default of `1.0`
        * If method="kde" and spherical=`False`, :obj:`scipy.stats.gaussian_kde`

    **kwargs
        Alternative way to specify `function_kws`


    Raises
    ------
    NotImplementedError
        If method given is not 'histogram' or 'kde'

    Returns
    -------
    xarray.DataArray
        Track density as a 2D map.

    """
    if function_kws is None:
        function_kws = {}

    function_kws = {**function_kws, **kwargs}

    if not spherical:
        # Account for cell area differences
        warnings.warn(
            "By default density does not take into account the spherical geometry of"
            "the Earth. Set spherical=True to account for this",
            stacklevel=2,
        )

    # Define coordinates for mapping
    if lon_range is None:
        lon_range = (-180, 180) if lon.min() < 0 or method == "line" else (0, 360)

    x_edge = np.arange(lon_range[0], lon_range[1] + bin_size, bin_size)
    y_edge = np.arange(lat_range[0], lat_range[1] + bin_size, bin_size)
    x_mid, y_mid = (x_edge[1:] + x_edge[:-1]) / 2, (y_edge[1:] + y_edge[:-1]) / 2

    # Compute density
    if method == "histogram":
        h = _histogram(lon, lat, x_edge, y_edge, function_kws)

        if spherical:
            h = h / _area(x_edge, y_edge)

    elif method == "line":
        if track_id is None:
            msg = "track_id must be set to calculate density using line intersections"
            raise ValueError(msg)

        h = _line_intersections(lon, lat, track_id, x_edge, y_edge, function_kws)

        if spherical:
            h = h / _area(x_edge, y_edge)

    elif method == "kde":
        if spherical:
            h = _spherical_kde(lon, lat, x_mid, y_mid, function_kws)

            # Normalise so that the area integral is the number of points
            h = h * len(lon) / (h * _area(x_edge, y_edge)).sum()
        else:
            h = _kde(lon, lat, x_mid, y_mid, function_kws)
    else:
        msg = f"Method {method} not implemented yet. Use one 'histogram', 'kde'"
        raise NotImplementedError(msg)

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
        return da.isel(lon=slice(idx_lon[0], idx_lon[-1] + 1))

    return da


def _histogram(lon, lat, x_edge, y_edge, function_kws):
    # Compute 2D histogram with numpy
    h, _x, _y = np.histogram2d(lon, lat, bins=[x_edge, y_edge], **function_kws)
    return h.T  # Transpose result


def _line_intersections(lon, lat, track_id, x_edge, y_edge, function_kws):
    tracks_df = to_geodataframe(lon, lat, track_id)

    # Create a geodataframe containing the grid as a set of boxes
    # Retain the indices to use in the histogram for counting
    x_indices, y_indices, geometries = [], [], []
    for y_idx in range(len(y_edge) - 1):
        for x_idx in range(len(x_edge) - 1):
            geometries.append(
                shapely.box(
                    x_edge[x_idx], y_edge[y_idx], x_edge[x_idx + 1], y_edge[y_idx + 1]
                )
            )
            x_indices.append(x_idx)
            y_indices.append(y_idx)

    grid = gpd.GeoDataFrame(
        dict(x_idx=x_indices, y_idx=y_indices, geometry=geometries), crs=Geodetic()
    )

    # A GeoDataFrame with one line for each track-gridbox intersection
    result = gpd.tools.sjoin(tracks_df, grid, predicate="intersects")

    # Convert to histogram using the saved gridbox indices
    h, _, _ = np.histogram2d(
        result.y_idx,
        result.x_idx,
        bins=[np.arange(-0.5, len(y_edge) - 1), np.arange(-0.5, len(x_edge) - 1)],
        **function_kws,
    )

    return h


def _kde(lon, lat, x_mid, y_mid, function_kws):
    # engineer positions array for kernel estimation computation
    positions = np.reshape(np.meshgrid(x_mid, y_mid), (2, len(x_mid) * len(y_mid)))
    # Compute kernel density estimate
    kernel = gaussian_kde([lon, lat], **function_kws)
    # Evaluation kernel along positions
    h = np.reshape(kernel(positions), (len(y_mid), len(x_mid)))

    # Normalize so that H integrates to the total number of points
    return h * len(lon) / h.sum()


def _spherical_kde(lon, lat, x_mid, y_mid, function_kws):
    if "bandwidth" not in function_kws:
        function_kws["bandwidth"] = "scott"
    if "metric" not in function_kws:
        function_kws["metric"] = "haversine"
    # engineer positions array for kernel estimation computation
    x_grid, y_grid = np.meshgrid(x_mid, y_mid)
    grid_positions = np.deg2rad(np.array([y_grid.flatten(), x_grid.flatten()]).T)
    track_positions = np.deg2rad([lat, lon]).T

    # Compute kernel density estimate
    kde = KernelDensity(**function_kws).fit(track_positions)

    # Evaluation kernel along positions
    return np.exp(kde.score_samples(grid_positions)).reshape(len(y_mid), len(x_mid))
