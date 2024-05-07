"""
Utils related to storm category
"""

import numpy as np
import xarray as xr

# TODO : More generic "get category" version


def categorise(variable, thresholds):

    categories = np.zeros_like(variable) * np.nan
    for category, threshold in thresholds.items():
        categories[(variable < threshold) & (np.isnan(categories))] = category

    return categories


_wind_thresholds = {
    "Saffir-Simpson": {-1: 16, 0: 29, 1: 38, 2: 44, 3: 52, 4: 63, 5: np.inf}
}


def get_sshs_cat(wind):  # TODO : Manage units properly (with pint?)
    """
    Function to determine the Saffir-Simpson Hurricane Scale (SSHS) category.

    Parameters
    ----------
    wind : xr.DataArray
        10-minutes averaged 10m wind in m/s

    Returns
    -------
    xarray.DataArray
        The category series.
        You can append it to your tracks by running tracks["sshs"] = get_sshs_cat(tracks.wind)
    """

    sshs = np.where(wind <= 16, -1, None)
    sshs = np.where((sshs == None) & (wind < 29), 0, sshs)
    sshs = np.where((sshs == None) & (wind < 38), 1, sshs)
    sshs = np.where((sshs == None) & (wind < 44), 2, sshs)
    sshs = np.where((sshs == None) & (wind < 52), 3, sshs)
    sshs = np.where((sshs == None) & (wind < 63), 4, sshs)
    sshs = np.where((sshs == None) & (~np.isnan(wind)), 5, sshs)
    sshs = np.where(sshs == None, np.nan, sshs)
    return xr.DataArray(sshs, dims="obs", coords={"obs": wind.obs})


_slp_thresholds = {
    "Simpson": [990, 980, 970, 965, 945, 920],
    "Klotzbach": [1005, 990, 975, 960, 945, 925],
}


def get_pressure_cat(p, convention="Klotzbach"):
    """
    Determine the pressure category according to selected convention.

    Parameters
    ----------
    slp : xr.DataArray
        Minimum Sea-level Pressure series in hPa
    convention : str
        Name of the classification convention you want to use.
            * Klotzbach (default)
            * Simpson

    Returns
    -------
    xarray.DataArray
        The category series.
        You can append it to your tracks by running tracks["cat"] = get_pressure_cat(tracks.slp)

    """

    p0, p1, p2, p3, p4, p5 = _slp_thresholds[convention]
    cat = np.where(p > p0, -1, None)
    cat = np.where((cat == None) & (p >= p1), 0, cat)
    cat = np.where((cat == None) & (p >= p2), 1, cat)
    cat = np.where((cat == None) & (p >= p3), 2, cat)
    cat = np.where((cat == None) & (p >= p4), 3, cat)
    cat = np.where((cat == None) & (p >= p5), 4, cat)
    cat = np.where((cat == None) & (~np.isnan(p)), 5, cat)
    cat = np.where(cat == None, np.nan, cat)
    return xr.DataArray(cat, dims="obs", coords={"obs": p.obs})
