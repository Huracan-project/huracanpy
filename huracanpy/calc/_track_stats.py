"""
Module containing functions to compute track statistics
"""

import numpy as np
import pandas as pd


def track_duration(time, track_ids):
    """
    Compute the duration of each track

    Parameters
    ----------
    time : xarray.DataArray
    track_ids : array_like

    Returns
    -------
    xarray.DataArray
        Duration of each track (in hours)

    """
    duration = (
        time.groupby(track_ids).map(lambda x: x.max() - x.min()).rename("duration")
    )
    duration = duration / np.timedelta64(1, "h")
    duration.attrs["units"] = "hours"
    return duration


def gen_vals(tracks, time, track_id):
    """
    Shows the attributes for the genesis point of each track

    Parameters
    ----------
    tracks : xarray.Dataset
    time : array_like
    track_id : xarray.DataArray

    Returns
    -------
    xarray.Dataset
        Dataset containing only genesis points, with track_id as index.

    """
    # It is 470 times much faster to switch to a dataframe...
    # Use the sortby/groupby with pandas to find the relevant indices in the original
    # Dataset by passing an index (named idx to not clash with "index")
    df = pd.DataFrame(
        data=dict(
            idx=np.arange(len(track_id)),
            time=np.array(time),
            track_id=np.array(track_id),
        )
    )
    idx = np.array(df.sort_values("time").groupby("track_id").first().idx)

    # Could check that track_id is 1d, but the function would already have failed by now
    # if not
    dim = track_id.dims[0]
    tracks = tracks.isel(**{dim: idx})

    # Promote track_id to a coordinate and remove record
    return tracks.assign_coords(**{track_id.name: tracks[track_id.name]}).swap_dims(
        **{dim: track_id.name}
    )


def apex_vals(tracks, variable, track_id, stat="max"):
    """
    Shows the attribute for the extremum point of each track

    Parameters
    ----------
    tracks : xarray.DataSet
    variable : array_like
        The extremum variable
    track_id : xarray.DataArray
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

    # It is 350 times much faster to switch to a dataframe.
    # Use the same trick as with gen_vals
    df = pd.DataFrame(
        data=dict(
            idx=np.arange(len(variable)),
            var=np.array(variable),
            track_id=np.array(track_id),
        )
    )
    idx = np.array(df.sort_values("var", ascending=asc).groupby("track_id").first().idx)

    dim = track_id.dims[0]
    tracks = tracks.isel(**{dim: idx})

    # Promote track_id to a coordinate and remove record
    return tracks.assign_coords(**{track_id.name: tracks[track_id.name]}).swap_dims(
        **{dim: track_id.name}
    )
