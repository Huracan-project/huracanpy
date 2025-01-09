"""
Module containing functions to compute lifecycle stage
"""

import pandas as pd


def time_from_genesis(time, track_ids):
    """
    Output the time since genesis for each TC point.

    Parameters
    ----------
    time: xr.DataArray
    track_ids: xr.DataArray

    Returns
    -------
    xarray.DataArray
        The time_from_genesis series.
        You can append it to your tracks by running

        >>> tracks["time_from_genesis"] = time_from_genesis(tracks.time, tracks.track_id)

    """
    data_df = pd.DataFrame({"time": time, "track_id": track_ids})
    data_df = data_df.merge(
        data_df.groupby("track_id").time.min(),
        on="track_id",
        suffixes=["_actual", "_gen"],
    )
    time_from_start = data_df.time_actual - data_df.time_gen
    return (
        time_from_start.to_xarray()
        .rename({"index": track_ids.dims[0]})
        .drop(track_ids.dims[0])
        .rename("time_from_genesis")
    )


def time_from_apex(time, track_ids, intensity_var, stat="max"):
    """The time relative to a maxima/minima in a given variable for each individual
    track

    >>> time_from_apex(tracks.time, tracks.track_id, tracks.wind, stat="max")

    Parameters
    ----------
    time : array_like
    track_ids : xarray.DataArray
    intensity_var : array_like
    stat : str, optional
        Take either the maxima ("max") or minima ("min") of `intensity_var`. Default is
        "max"

    Returns
    -------
    xarray.DataArray

    """
    if stat == "max":
        asc = False
    elif stat == "min":
        asc = True
    else:
        raise NotImplementedError("stat not recognized. Please use one of {min, max}")
    data_df = pd.DataFrame(
        {"time": time, "track_id": track_ids, "intensity": intensity_var}
    )
    extr = data_df.sort_values("intensity", ascending=asc).groupby("track_id").first()
    data_df = data_df.merge(extr, on="track_id", suffixes=["_actual", "_extr"])
    time_from_extr = data_df.time_actual - data_df.time_extr
    return (
        time_from_extr.to_xarray()
        .rename({"index": track_ids.dims[0]})
        .drop(track_ids.dims[0])
        .rename("time_from_extremum")
    )
