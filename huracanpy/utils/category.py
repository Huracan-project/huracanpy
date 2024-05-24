"""
Utils related to storm category
"""

import numpy as np
import xarray as xr
import pandas as pd


def categorise(variable, thresholds):
    categories = np.zeros_like(variable) * np.nan
    for category, threshold in thresholds.items():
        categories[(variable < threshold) & (np.isnan(categories))] = category

    return categories


_thresholds = {
    "Klotzbach": {5: 925, 4: 945, 3: 960, 2: 975, 1: 990, 0: 1005, -1: np.inf},
    "Simpson": {5: 920, 4: 945, 3: 965, 2: 970, 1: 980, 0: 990, -1: np.inf},
    "Saffir-Simpson": {-1: 16, 0: 29, 1: 38, 2: 44, 3: 52, 4: 63, 5: np.inf},
}

categorize = categorise  # American spelling


def get_sshs_cat(wind):  # TODO : Manage units properly (with pint?)
    """
    Function to determine the Saffir-Simpson Hurricane Scale (SSHS) category.

    Parameters
    ----------
    wind : xarray.DataArray
        10-minutes averaged 10m wind in m/s

    Returns
    -------
    xarray.DataArray
        The category series.
        You can append it to your tracks by running tracks["sshs"] = get_sshs_cat(tracks.wind)
    """

    sshs = categorise(wind, _thresholds["Saffir-Simpson"])
    return xr.DataArray(sshs, dims="record", coords={"record": wind.record})


def get_pressure_cat(slp, convention="Klotzbach"):
    """
    Determine the pressure category according to selected convention.

    Parameters
    ----------
    slp : xarray.DataArray
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

    if slp.min() > 10000:
        print(
            "Caution, pressure are likely in Pa, they are being converted to hPa for categorization"
        )
        slp = slp / 100

    cat = categorise(slp, thresholds=_thresholds[convention])
    return xr.DataArray(cat, dims="record", coords={"record": slp.record})


# [Stella] Leaving that here as an alternative method memo if we encounter performance issues.
def categorize_alt(var, bins, labels=None):
    """
    Provides category according to provided bins and labels

    Parameters
    ----------
    var : xarray.DataArray
        The variable to categorize
    bins : array_like
        bins boundaries
    labels : array_like, optional
        Name of the categories. len(labels) = len(bins) -1

    Returns
    -------
    xarray.DataArray
        The category series.
        You can append it to your tracks by running tracks["cat"] = categorize(tracks.var, bins)
    """
    cat = pd.cut(var, bins, labels=labels)
    return xr.DataArray(cat, dims="record", coords={"record": var.record})
