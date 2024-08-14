"""
Module containing functions to compute ACE
"""

import xarray as xr
import pint
from metpy.xarray import preprocess_and_wrap
from metpy.units import units


def ace_by_point(wind, threshold=34 * units("knots"), wind_units="m s-1"):
    """Calculate accumulate cyclone energy (ACE) for each individual point

    Parameters
    ----------
    wind : array_like
        Maximum velocity of a tropical cyclone
    threshold : scalar, default=34 knots
        ACE is set to zero below this threshold wind speed. The default argument is in
        knots. To pass an argument with units, use :py:mod:`metpy.units`, otherwise any
        non-default argument will be assumed to have the units of "wind_units" which is
        "m s-1" by default.
    wind_units : str, default="m s-1"
        If the units of wind are not specified in the attributes then the function will
        assume it is in these units before converting to knots

    Returns
    -------
    array_like
        The ACE at each point in wind

    """
    ace_values = _ace_by_point(wind, threshold, wind_units)

    # The return value has units so stays as a pint.Quantity
    # This can be annoying if you still want to do other things with the array
    # Metpy dequantify keeps the units as an attribute so it can still be used later
    # TODO - extend preprocess_and_wrap to include this if it is needed for more
    #  functions
    if isinstance(ace_values, xr.DataArray) and isinstance(
        ace_values.data, pint.Quantity
    ):
        ace_values = ace_values.metpy.dequantify()

    return ace_values


@preprocess_and_wrap(wrap_like="wind")
def _ace_by_point(wind, threshold=34 * units("knots"), wind_units="m s-1"):
    if not isinstance(wind, pint.Quantity) or wind.unitless:
        wind = wind * units(wind_units)
    wind = wind.to(units("knots"))

    if not isinstance(threshold, pint.Quantity) or threshold.unitless:
        threshold = threshold * units(wind_units)

    wind[wind < threshold] = 0 * units("knots")

    ace_values = (wind**2.0) * 1e-4

    return ace_values
