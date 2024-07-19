"""
Module containing functions to compute track statistics
"""

import xarray as xr
import pint
from metpy.xarray import preprocess_and_wrap
from metpy.units import units


def ace(wind, threshold=34 * units("knots"), wind_units="m s-1"):
    r"""Calculate accumulate cyclone energy (ACE) for each track

    .. math:: \mathrm{ACE} = 10^{-4} \sum v_\mathrm{max}^2 \quad (v_\mathrm{max} \ge 34 \mathrm{kn})

    Parameters
    ----------
    wind : array_like
        Maximum velocity of a tropical cyclone. Must also have an associated "track_id"
        coordinate to allow summing for each track
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
        The ACE for each track in wind

    """
    ace_ = ace_by_point(wind, threshold, wind_units)

    ace_by_storm = ace_.groupby("track_id").sum()

    return ace_by_storm


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
        ace_values.values, pint.Quantity
    ):
        ace_values = ace_values.metpy.dequantify()

    return ace_values


@preprocess_and_wrap(wrap_like="wind")
def _ace_by_point(wind, threshold=34 * units("knots"), wind_units="m s-1"):
    if not isinstance(ace, pint.Quantity) or wind.unitless:
        wind = wind * units(wind_units)
    wind = wind.to(units("knots"))

    if not isinstance(threshold, pint.Quantity) or threshold.unitless:
        threshold = threshold * units(wind_units)

    wind[wind < threshold] = 0 * units("knots")

    ace_values = (wind**2.0) * 1e-4

    return ace_values


def duration(tracks):
    """
    Compute the duration of each track

    Parameters
    ----------
    tracks : xarray.Dataset

    Returns
    -------
    xarray.DataArray
        Duration of each track

    """
    duration = (
        tracks.groupby("track_id")
        .map(lambda x: x.time.max() - x.time.min())
        .rename("duration")
    )
    duration = (duration * 1e-9 / 3600).astype(float)
    duration.attrs["units"] = "h"
    return duration


def gen_vals(tracks):
    """
    Shows the attributes for the genesis point of each track

    Parameters
    ----------
    tracks : xarray.DataSet

    Returns
    -------
    xarray.Dataset
        Dataset containing only genesis points, with track_id as index.

    """

    return tracks.sortby("time").groupby("track_id").first()


def extremum_vals(tracks, varname, stat="max"):
    """
    Shows the attribute for the extremum point of each track

    Parameters
    ----------
    tracks : xarray.DataSet
    var : str
        The extremum variable
    stat : str, optional
        Type of extremum. Can be "min" or "max". The default is "max".

    Raises
    ------
    NotImplementedError
        If another value than "min" and "max" is given to stat

    Returns
    -------
    xarray.Dataset
        Dataset containing only extremum points, with track_id as index.

    """

    # tracks will be sorted along var and then the first line of each track_id will be used
    # asc determines whether the sorting must be ascending (True) or descending (False)
    if stat == "max":
        asc = False
    elif stat == "min":
        asc = True
    else:
        raise NotImplementedError("stat not recognized. Please use one of {min, max}")

    return tracks.sortby(varname, ascending=asc).groupby("track_id").first()
