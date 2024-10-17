"""
Utils related to time
"""

import pandas as pd
import numpy as np
from metpy.xarray import preprocess_and_wrap

from .geography import get_hemisphere


def get_time_components(time):
    """
    Expand the time variable into year/month/day/hour

    Parameters
    ----------
    time : xr.DataArray
        The time time series.

    Returns
    -------
    year, month, day, hour : xr.DataArrays

    """
    year = time.dt.year
    month = time.dt.month
    day = time.dt.day
    hour = time.dt.hour
    return year, month, day, hour


@preprocess_and_wrap(wrap_like="track_id")
def get_season(track_id, lat, time, convention="short"):
    """


    Parameters
    ----------
    track_id : xarray.DataArray
    lat : xarray.DataArray
    time : xarray.DataArray
    convention : str
        * 'short' : In the southern hemisphere, the season n corresponds to July n-1 to June n
        * 'long' : In the southern hemisphere, the season from July n-1 to June n is named "(n-1)n"

    Raises
    ------
    NotImplementedError
        If convention given is not 'short' or 'long'

    Returns
    -------
    xarray.DataArray
        The season series.
        You can append it to your tracks by running tracks["season"] = get_season(tracks.track_id, tracks.lat, tracks.time)
    """

    # Derive values
    hemi = get_hemisphere(lat)

    time = pd.to_datetime(time)
    year = time.year
    month = time.month
    # Store in a dataframe
    df = pd.DataFrame(
        {"hemi": hemi, "year": year, "month": month, "track_id": track_id}
    )
    # Most frequent year, month and hemisphere for each track
    # Grouping is done to avoid labelling differently points in a track that might cross hemisphere or seasons.
    group = df.groupby("track_id")[["year", "month", "hemi"]].agg(
        lambda x: pd.Series.mode(x)[0]
    )

    # Assign season
    if convention == "short":
        season = np.where(group.hemi == "N", group.year, np.nan)
        season = np.where(
            (group.hemi == "S") & (group.month >= 7), group.year + 1, season
        )
        season = np.where((group.hemi == "S") & (group.month <= 6), group.year, season)
    elif convention == "long":
        season = np.where(group.hemi == "N", group.year.astype(str), np.nan)
        season = np.where(
            (group.hemi == "S") & (group.month >= 7),
            group.year.astype(str) + (group.year + 1).astype(str),
            season,
        )
        season = np.where(
            (group.hemi == "S") & (group.month <= 6),
            (group.year - 1).astype(str) + group.year.astype(str),
            season,
        )
    else:
        raise NotImplementedError("Convention not recognized")

    group["season"] = season
    df = df.merge(group[["season"]], on="track_id")

    return df.season.values
