"""
Module containing functions to compute track statistics
"""


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
    return (
        tracks.groupby("track_id")
        .map(lambda x: x.time.max() - x.time.min())
        .rename("duration")
    )


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
