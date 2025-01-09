"""
Utils related to time
"""

import pandas as pd
import numpy as np
from metpy.xarray import preprocess_and_wrap

from ._geography import hemisphere


def time_components(time, components=("year", "month", "day", "hour")):
    """
    Expand the time variable into year/month/day/hour

    Parameters
    ----------
    time : xr.DataArray
        The time series
    components : iterable[str], optional
        The time components of `time` to return. Can include any valid attribute of
        `xarray.core.accessor_dt.DatetimeAccessor`. Default is year, month, day, hour

    Returns
    -------
    list[xarray.DataArray]
        A DataArray for each requested time component

    """
    return [getattr(time.dt, component) for component in components]


@preprocess_and_wrap(wrap_like="track_id")
def season(track_id, lat, time, convention="tc-short"):
    """Determine the cyclone season for each track

    Parameters
    ----------
    track_id : xarray.DataArray
    lat : xarray.DataArray
    time : xarray.DataArray
    convention : str
        * 'tc-short' : In the Northern hemisphere, the season is the same as calendar year.
        In the southern hemisphere, the season n corresponds to July n-1 to June n
        * 'tc-long' : In the Northern hemisphere, the season is the same as calendar year.
        In the southern hemisphere, the season from July n-1 to June n is named "(n-1)n"

    Raises
    ------
    NotImplementedError
        If convention given is not 'tc-short' or 'tc-long'

    Returns
    -------
    xarray.DataArray
        The season series.
        You can append it to your tracks by running tracks["season"] = get_season(tracks.track_id, tracks.lat, tracks.time)
    """

    # Derive values
    hemi = hemisphere(lat)

    try:
        time = pd.to_datetime(time)
        year = time.year
        month = time.month
    except TypeError:
        # Fix for cftime
        year = np.array([t.year for t in time])
        month = np.array([t.month for t in time])

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
    if convention == "tc-short":
        season = np.where(group.hemi == "N", group.year, np.nan)
        season = np.where(
            (group.hemi == "S") & (group.month >= 7), group.year + 1, season
        )
        season = np.where((group.hemi == "S") & (group.month <= 6), group.year, season)
    elif convention == "tc-long":
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
