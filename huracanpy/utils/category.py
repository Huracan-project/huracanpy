"""
Utils related to storm category
"""

import numpy as np
import pint
import xarray as xr
import pandas as pd

from metpy.xarray import preprocess_and_wrap
from metpy.units import units


@preprocess_and_wrap(wrap_like="variable")
def categorize(variable, bins=None, labels=None, convention=None, variable_units=None):
    """Calculate a generic category from a variable and a set of thresholds

    Parameters
    ----------
    variable : array_like
        The variable to be categorized
    bins : array_like
        Bounds for the categories, including upper and lower bounds
    labels : array_like
        Name of the categories. len(labels) = len(bins) -1
    convention : str
        * Klotzbach
        * Simpson
        * Saffir-Simpson
    variable_units : str
        The units of the input variable. Only needs to be specified if they are
        different to the units of the bins and they are not already in the attributes of
        the variable.

    Returns
    -------
    numpy.ndarray
        The category label for each value in the input variable

    """
    if bins is None and labels is None:
        if convention is None:
            raise ValueError("Must specify either bins/labels or convention")
        else:
            bins = _thresholds[convention]["bins"]
            labels = _thresholds[convention]["labels"]

    if not isinstance(variable, pint.Quantity) or variable.unitless:
        if variable_units is None and isinstance(bins, pint.Quantity):
            variable_units = str(bins.units)
        variable = variable * units(variable_units)

    categories = np.zeros_like(variable) * np.nan
    for n, label in enumerate(labels):
        categories[(bins[n] < variable) & (variable <= bins[n + 1])] = label

    return categories


_thresholds = {
    "Klotzbach": dict(
        bins=np.array([-np.inf, 925, 945, 960, 975, 990, 1005, np.inf]) * units("hPa"),
        labels=[5, 4, 3, 2, 1, 0, -1],
    ),
    "Simpson": dict(
        bins=np.array([-np.inf, 920, 945, 965, 970, 980, 990, np.inf]) * units("hPa"),
        labels=[5, 4, 3, 2, 1, 0, -1],
    ),
    "Saffir-Simpson": dict(
        bins=np.array([-np.inf, 16, 29, 38, 44, 52, 63, np.inf]) * units("m s-1"),
        labels=[-1, 0, 1, 2, 3, 4, 5],
    ),
}


@preprocess_and_wrap(wrap_like="wind")
def get_sshs_cat(wind, convention="Saffir-Simpson", wind_units="m s-1"):
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
    return categorize(wind, convention=convention, variable_units=wind_units)


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

    return categorize(slp, convention=convention, variable_units=slp_units)


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
