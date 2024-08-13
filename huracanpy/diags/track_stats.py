"""
Module containing functions to compute track statistics
"""

# import xarray as xr
# import pint
# from metpy.xarray import preprocess_and_wrap
from metpy.units import units

from huracanpy.utils.ace import ace_by_point


def ace_by_track(
    tracks,
    wind,
    threshold=34 * units("knots"),
    wind_units="m s-1",
    keep_ace_by_point=False,
):
    r"""Calculate accumulate cyclone energy (ACE) for each track

    .. math:: \mathrm{ACE} = 10^{-4} \sum v_\mathrm{max}^2 \quad (v_\mathrm{max} \ge 34 \mathrm{kn})

    Parameters
    ----------
    tracks : xarray.Dataset
        Full dataset of tracks data. Must have an associated "track_id" variable to
        allow summing for each track
    wind : array_like
        Maximum velocity of a tropical cyclone associated with the tracks dataset
    threshold : scalar, default=34 knots
        ACE is set to zero below this threshold wind speed. The default argument is in
        knots. To pass an argument with units, use :py:mod:`metpy.units`, otherwise any
        non-default argument will be assumed to have the units of "wind_units" which is
        "m s-1" by default.
    wind_units : str, default="m s-1"
        If the units of wind are not specified in the attributes then the function will
        assume it is in these units before converting to knots
    keep_ace_by_point : bool, default=False
        If True the ACE calculated from each point of the input wind is saved in the
        input tracks dataset as "ace"

    Returns
    -------
    array_like
        The ACE for each track in wind

    """
    tracks["ace"] = ace_by_point(wind, threshold, wind_units)

    ace_by_storm = tracks.groupby("track_id").map(lambda x: x.ace.sum())

    if not keep_ace_by_point:
        del tracks["ace"]

    return ace_by_storm


def duration(time, track_ids):
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
        time.groupby(track_ids).map(lambda x: x.max() - x.min()).rename("duration")
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
