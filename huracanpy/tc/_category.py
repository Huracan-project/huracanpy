"""
Module with function to compute TC-specific categories
"""

import warnings

from metpy.units import units
from metpy.xarray import preprocess_and_wrap

from ..info import category
from .._metpy import dequantify_results, validate_units
from ._conventions import _thresholds


def saffir_simpson_category(wind, convention="Saffir-Simpson", wind_units="m s-1"):
    """
    Function to determine the Saffir-Simpson Hurricane Scale (SSHS) category.

    Parameters
    ----------
    wind : array_like
        10-minutes averaged 10m wind in m/s

    convention : str

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
        bins=_thresholds[convention]["bins"],
        labels=_thresholds[convention]["labels"],
        variable_units=wind_units,
    )


@dequantify_results
@preprocess_and_wrap(wrap_like="slp")
def pressure_category(slp, convention="Klotzbach", slp_units=None):
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
        You can append it to your tracks by running
        tracks["cat"] = get_pressure_cat(tracks.slp)

    """
    # Don't automatically switch units to Pa if they have been explicitly set to Pa,
    # even if they seem wrong
    if (not isinstance(slp, units.Quantity) or slp.unitless) and slp_units is None:
        if slp.magnitude.min() > 10000:
            warnings.warn(
                "Caution, pressure are likely in Pa, they are being converted to hPa "
                "for categorization. In future specify the units explicitly by passing "
                'slp_units="Pa" to this function or setting '
                'slp.attrs["units"] = "Pa"'
            )
            slp_units = "Pa"
        else:
            slp_units = "hPa"

    slp = validate_units(slp, expected_units=slp_units)

    return category(
        slp,
        bins=_thresholds[convention]["bins"],
        labels=_thresholds[convention]["labels"],
        variable_units=slp_units,
    )
