"""
Utils related to storm category
"""

import numpy as np
import pint
import xarray as xr
import pandas as pd

from metpy.xarray import preprocess_and_wrap
from metpy.units import units


def categorize(variable, thresholds):
    """Calculate a generic category from a variable and a set of thresholds

    Parameters
    ----------
    variable : array_like
        The variable to be categorized
    thresholds : dict
        Mapping of category value to lower bound

    Returns
    -------
    numpy.ndarray
        The category value for each value in the input variable

    """
    categories = np.zeros_like(variable) * np.nan
    for category, threshold in thresholds.items():
        categories[(variable < threshold) & (np.isnan(categories))] = category

    return categories


_thresholds = {
    "Klotzbach": {5: 925, 4: 945, 3: 960, 2: 975, 1: 990, 0: 1005, -1: np.inf},
    "Simpson": {5: 920, 4: 945, 3: 965, 2: 970, 1: 980, 0: 990, -1: np.inf},
    "Saffir-Simpson": {-1: 16, 0: 29, 1: 38, 2: 44, 3: 52, 4: 63, 5: np.inf},
}


# There is probably a better way of defining the thresholds
for name, threshold_unit in [
    ("Klotzbach", units("hPa")),
    ("Simpson", units("hPa")),
    ("Saffir-Simpson", units("m s-1")),
]:
    for threshold in _thresholds[name]:
        _thresholds[name][threshold] = _thresholds[name][threshold] * threshold_unit


@preprocess_and_wrap(wrap_like="wind")
def get_sshs_cat(wind, wind_units="m s-1"):
    """
    Function to determine the Saffir-Simpson Hurricane Scale (SSHS) category.

    Parameters
    ----------
    wind : xarray.DataArray
        10-minutes averaged 10m wind in m/s

    wind_units : str, default="m s-1"
        The units of the input array if they are not already provided by the attributes

    Returns
    -------
    xarray.DataArray
        The category series.
        You can append it to your tracks by running tracks["sshs"] = get_sshs_cat(tracks.wind)
    """
    if not isinstance(wind, pint.Quantity) or wind.unitless:
        wind = wind * units(wind_units)

    return categorize(wind, _thresholds["Saffir-Simpson"])


@preprocess_and_wrap(wrap_like="slp")
def get_pressure_cat(slp, convention="Klotzbach", slp_units="hPa"):
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
    slp_units : str, default="hPa"
        The units of the input array if they are not already provided by the attributes

    Returns
    -------
    xarray.DataArray
        The category series.
        You can append it to your tracks by running tracks["cat"] = get_pressure_cat(tracks.slp)

    """
    if not isinstance(slp, pint.Quantity) or slp.unitless:
        if slp.min() > 10000:
            print(
                "Caution, pressure are likely in Pa, they are being converted to hPa "
                "for categorization. In future specify the units explicitly by passing "
                'slp_units="hPa" to this function or setting '
                'slp.attrs["units"] = "hPa"'
            )
            slp = slp / 100

        slp = slp * units(slp_units)

    return categorize(slp, thresholds=_thresholds[convention])


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
