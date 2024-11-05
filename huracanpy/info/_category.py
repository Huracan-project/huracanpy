"""
Function to categorise
"""

import numpy as np
import pint
import xarray as xr
import pandas as pd

from metpy.xarray import preprocess_and_wrap
from metpy.units import units


@preprocess_and_wrap(wrap_like="variable")
def get_category(variable, bins, labels=None, variable_units=None):
    """Calculate a generic category from a variable and a set of thresholds

    Parameters
    ----------
    variable : array_like
        The variable to be categorized
    bins : array_like
        Bounds for the categories, including upper and lower bounds
    labels : array_like
        Name of the categories. len(labels) = len(bins) -1
    variable_units : str
        The units of the input variable. Only needs to be specified if they are
        different to the units of the bins and they are not already in the attributes of
        the variable.

    Returns
    -------
    numpy.ndarray
        The category label for each value in the input variable

    """

    if labels is None:
        print(
            "labels not provided, categories will be named from 1 to n in the order of the provided bins"
        )
        labels = [str(i) for i in range(len(bins) - 1)]

    if not isinstance(variable, pint.Quantity) or variable.unitless:
        if variable_units is None and isinstance(bins, pint.Quantity):
            variable_units = str(bins.units)
        variable = variable * units(variable_units)

    categories = np.zeros_like(variable) * np.nan
    for n, label in enumerate(labels):
        categories[(bins[n] < variable) & (variable <= bins[n + 1])] = label

    return categories


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
