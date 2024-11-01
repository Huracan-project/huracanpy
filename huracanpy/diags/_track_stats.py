"""
Module containing functions to compute track statistics
"""


def get_track_duration(time, track_ids):
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


def get_gen_vals(tracks, time_name="time", track_id_name="track_id"):
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


def get_apex_vals(tracks, varname, stat="max", track_id_name="track_id"):
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
