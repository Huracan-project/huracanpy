"""
Module containing functions to compute track statistics
"""

from metpy.units import units

from huracanpy.utils.ace import get_ace, get_pace


def ace_by_track(
    tracks,
    wind,
    threshold=34 * units("knots"),
    wind_units="m s-1",
    keep_ace_by_point=False,
    ace_varname="ace",
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
        input tracks dataset as `ace_varname`
    ace_varname : str, default="ace"
        The name to give the variable for ACE at each point added to the `tracks`
        dataset. Change this if you want to have a different variable name or want to
        avoid overwriting an existing variable in the dataset named `ace`

    Returns
    -------
    array_like
        The ACE for each track in wind

    """
    tracks[ace_varname] = get_ace(wind, threshold, wind_units)

    ace_by_storm = tracks.groupby("track_id").map(lambda x: x[ace_varname].sum())

    if not keep_ace_by_point:
        del tracks[ace_varname]

    return ace_by_storm


def pace_by_track(
    tracks,
    pressure,
    wind=None,
    model=None,
    threshold_wind=None,
    threshold_pressure=None,
    wind_units="m s-1",
    keep_pace_by_point=False,
    pace_varname="pace",
    **kwargs,
):
    """Calculate a pressure-based accumulated cyclone energy (PACE) for each individual
       point

    PACE is calculated the same way as ACE, but the wind is derived from fitting a
    pressure-wind relationship and calculating wind values from pressure using this fit

    Example
    -------
    This function can be called in two ways

    1. Pass the pressure and wind to fit a pressure-wind relationship to the data and
    then calculate pace from the winds derived from this fit

    >>> pace, pw_model = get_pace(pressure, wind)

    The default model to fit is a quadratic polynomial
    (:py:class:`numpy.polynomial.polynomial.Polynomial` with `deg=2`)

    2. Pass just the pressure and an already fit model to calculate the wind speeds from
    this model

    >>> pace, _ = get_pace(pressure, model=pw_model)

    Parameters
    ----------
    tracks : xarray.Dataset
    pressure : array_like
    wind : array_like, optional
    model : str, class, or object, optional
    threshold_wind : scalar, optional
    threshold_pressure : scalar, optional
    wind_units : str, default="m s-1"
    keep_pace_by_point : bool, default=False
        If True the PACE calculated from each point of the input wind is saved in the
        input tracks dataset as `pace_varname`
    pace_varname : str, default="pace"
    **kwargs

    Returns
    -------
    pace_values : array_like

    model : object

    """
    tracks[pace_varname], model = get_pace(
        pressure,
        wind=wind,
        model=model,
        threshold_wind=threshold_wind,
        threshold_pressure=threshold_pressure,
        wind_units=wind_units,
        **kwargs,
    )

    pace_by_storm = tracks.groupby("track_id").map(lambda x: x[pace_varname].sum())

    if not keep_pace_by_point:
        del tracks[pace_varname]

    return pace_by_storm, model


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


def gen_vals(tracks, time_name="time", track_id_name="track_id"):
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

    return (
        tracks.to_dataframe()
        .sort_values(time_name)
        .groupby(track_id_name)
        .first()
        .to_xarray()
    )  # It is 470 times much faster to switch to a dataframe...


def extremum_vals(tracks, varname, stat="max", track_id_name="track_id"):
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

    return (
        tracks.to_dataframe()
        .sort_values(varname, ascending=asc)
        .groupby(track_id_name)
        .first()
        .to_xarray()
    )  # It is 350 times much faster to switch to a dataframe..
