"""
Module with function to compute TC-specific categories
"""

import warnings

from metpy.units import units
from metpy.xarray import preprocess_and_wrap

from ..info import category
from .._metpy import dequantify_results, validate_units
from ._conventions import _thresholds


def saffir_simpson_category(wind, wind_units="m s-1", convention="10min"):
    """
    Determine the Saffir-Simpson Hurricane Scale (SSHS) category.

    Parameters
    ----------
    wind : array_like
        Maximum sustained wind speed. Averaging period depends on the convention used
        (see convention)

    wind_units : str, default="m s-1"
        The units of the input array if they are not already provided by the attributes

    convention : str, default="10min"
        The thresholds based on the averaging periods used for the wind. Pass one of
            * "10min" or "wmo
            * "1min" or "nhc"

        The default is the WMO convention of 10-minute sustained winds, which has lower
        thresholds than the alternative NHC convention of 1-minute sustained winds

    Returns
    -------
    array_like
        An array of integers ranging from -1 to 5. 1 to 5 is the Saffir-Simpson
        category, zero is tropical storm, and -1 is anything weaker
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
