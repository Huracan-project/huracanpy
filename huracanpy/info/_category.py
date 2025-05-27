"""
Function to categorise
"""

import warnings

import numpy as np
import pint
from pint.errors import UnitStrippedWarning
import pandas as pd

from metpy.xarray import preprocess_and_wrap
from metpy.units import units


@preprocess_and_wrap(wrap_like="variable")
def category(variable, bins, labels=None, variable_units=None):
    """Calculate a generic category from a variable and a set of thresholds

    >>> huracanpy.info.category(tracks.wind, bins = [0,10,20,30], labels = [1,2,3])

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

    # Account for one, both, or neither of the variable and bins having their units
    # specified
    if not isinstance(variable, pint.Quantity) or variable.unitless:
        # If variable has no units, but bins do, copy the units from the bins to the
        # variable. But if neither have units specified use the "variable_units" kwarg
        if variable_units is None and isinstance(bins, pint.Quantity):
            variable_units = str(bins.units)
        variable = variable * units(variable_units)

    # If bins has no units, copy the units to from the variable. Which may have already
    # been set by the "variable_units" kwarg
    if not isinstance(bins, pint.Quantity) or bins.unitless:
        bins = bins * variable.units

    # Make sure the units match, however they have been set
    bins = bins.to(variable.units)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UnitStrippedWarning,
            message="The unit of the quantity is stripped when downcasting to ndarray.",
        )

        result = np.array(pd.cut(variable, bins, labels=labels))

    return result
