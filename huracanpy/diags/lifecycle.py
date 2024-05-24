"""
Module containing functions to compute lifecycle stage
"""


def time_from_genesis(data):
    """
    Output the time since genesis for each TC point

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    xarray.DataArray
        The time_from_genesis series.
        You can append it to your tracks by running tracks["time_from_genesis"] = time_from_genesis(tracks)

    """
    data_df = data[["track_id", "time"]].to_dataframe()
    data_df = data_df.merge(
        data_df.groupby("track_id").time.min(),
        on="track_id",
        suffixes=["_actual", "_gen"],
    )
    time_from_start = data_df.time_actual - data_df.time_gen
    return (
        time_from_start.to_xarray().rename({"index": "obs"}).rename("time_from_genesis")
    )


def time_from_extremum(data, varname, stat=max):
    """

    Parameters
    ----------
    data
    varname
    stat

    Returns
    -------

    """
    if stat == "max":
        asc = False
    elif stat == "min":
        asc = True
    else:
        raise NotImplementedError("stat not recognized. Please use one of {min, max}")
    data_df = data[["track_id", "time", varname]].to_dataframe()
    extr = data_df.sort_values(varname, ascending=asc).groupby("track_id").first()
    data_df = data_df.merge(extr, on="track_id", suffixes=["_actual", "_extr"])
    time_from_extr = data_df.time_actual - data_df.time_extr
    return (
        time_from_extr.to_xarray().rename({"index": "obs"}).rename("time_from_extremum")
    )
