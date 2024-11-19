"""
Function to categorise
"""

import warnings

import numpy as np
import pint
import pandas as pd

from metpy.xarray import preprocess_and_wrap
from metpy.units import units


@preprocess_and_wrap(wrap_like="variable")
def category(variable, bins, labels=None, variable_units=None):
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
    array_like
        The category label for each value in the input variable

    """

    if labels is None:
        warnings.warn(
            "labels not provided, categories will be named from 1 to n in the order of"
            "the provided bins"
        )
        labels = [str(i) for i in range(len(bins) - 1)]

    if not isinstance(variable, pint.Quantity) or variable.unitless:
        if variable_units is None and isinstance(bins, pint.Quantity):
            variable_units = str(bins.units)
        variable = variable * units(variable_units)

    if not isinstance(bins, pint.Quantity) or bins.unitless:
        bins = bins * units(variable_units)

    bins = bins.to(variable.units)

    return np.array(pd.cut(variable, bins, labels=labels))
