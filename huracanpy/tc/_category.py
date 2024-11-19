"""
Module with function to compute TC-specific categories
"""

import warnings

import pint
from metpy.xarray import preprocess_and_wrap

from ..info import category
from .._metpy import dequantify_results
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
        You can append it to your tracks by running tracks["sshs"] = get_sshs_cat(tracks.wind)
    """
    return category(
        wind,
        bins=_thresholds[convention]["bins"],
        labels=_thresholds[convention]["labels"],
        variable_units=wind_units,
    )


@dequantify_results
@preprocess_and_wrap(wrap_like="slp")
def pressure_category(slp, convention="Klotzbach", slp_units="hPa"):
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
        if slp.min() > 10000 and slp_units == "hPa":
            warnings.warn(
                "Caution, pressure are likely in Pa, they are being converted to hPa "
                "for categorization. In future specify the units explicitly by passing "
                'slp_units="Pa" to this function or setting '
                'slp.attrs["units"] = "Pa"'
            )
            slp = slp / 100

    return category(
        slp,
        bins=_thresholds[convention]["bins"],
        labels=_thresholds[convention]["labels"],
        variable_units=slp_units,
    )
