"""
Function to categorise
"""

import warnings

import numpy as np
import pint
from pint.errors import UnitStrippedWarning
import pandas as pd

from metpy.xarray import preprocess_and_wrap


from ._conventions import _thresholds
from .._metpy import validate_units


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
    variable = validate_units(
        variable,
        # If variable has no units, but bins do, copy the units from the bins to the
        # variable. But if neither have units specified use the "variable_units" kwarg
        expected_units=str(bins.units)
        if variable_units is None and isinstance(bins, pint.Quantity)
        else variable_units,
    )

    # If bins has no units, copy the units to from the variable. Which may have already
    # been set by the "variable_units" kwarg
    bins = validate_units(bins, str(variable.units))

    # Make sure the units match, however they have been set
    bins = bins.to(variable.units)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UnitStrippedWarning,
            message="The unit of the quantity is stripped when downcasting to ndarray.",
        )

        result = np.asarray(pd.cut(variable, bins, labels=labels))

    return result


def beaufort_category(wind, wind_units="m s-1"):
    """Beaufort Wind Scale category

    Parameters
    ----------
    wind : array_like
        10-minutes averaged 10m wind

    wind_units : str, default="m s-1"
        The units of the input array if they are not already provided by the attributes

    Returns
    -------
    array_like
        The category series.
        You can append it to your tracks by running
        tracks["sshs"] = get_sshs_cat(tracks.wind)
    """
    return category(
        wind,
        bins=_thresholds["Beaufort"]["bins"],
        labels=_thresholds["Beaufort"]["labels"],
        variable_units=wind_units,
    )
