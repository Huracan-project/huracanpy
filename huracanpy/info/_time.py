"""
Utils related to time
"""

import warnings

from metpy.xarray import preprocess_and_wrap
import numpy as np
from pint.errors import UnitStrippedWarning
import pandas as pd

from ._geography import hemisphere


def timestep(time, track_id=None):
    """Infer the timestep given a set of times (and optionally track IDs)

    Parameters
    ----------
    time : array_like
    track_id : array_like

    Returns
    -------
    scalar
        The timestep inferred from time. Will be the same type as time

    """
    step = np.diff(time)

    if track_id is not None:
        # Ignore where the track_id changes
        track_id = np.asarray(track_id)
        step = step[track_id[1:] == track_id[:-1]]

    steps, counts = np.unique(step, return_counts=True)

    if len(steps) == 1:
        return steps[0]
    else:
        warnings.warn(
            "Found multiple different timesteps within the tracks\n"
            + ", ".join([str(step) for step in steps])
            + "\n"
            + "Returning the most common one."
        )
        return steps[counts.argmax()]


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

        * 'tc-short' : In the Northern hemisphere, the season is the same as calendar
          year. In the southern hemisphere, the season n corresponds to July n-1 to
          June n
        * 'tc-long' : In the Northern hemisphere, the season is the same as calendar
          year. In the southern hemisphere, the season from July n-1 to June n is named
          "(n-1)n"

    Raises
    ------
    NotImplementedError
        If convention given is not 'tc-short' or 'tc-long'

    Returns
    -------
    xarray.DataArray
        The season series. You can append it to your tracks by running
        tracks["season"] = get_season(tracks.track_id, tracks.lat, tracks.time)
    """

    # Derive values
    hemi = hemisphere(lat)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UnitStrippedWarning,
                message="The unit of the quantity is stripped",
            )
            time = pd.to_datetime(time)
        year = time.year
        month = time.month
    except TypeError:
        # Fix for cftime
        year = np.asarray([t.year for t in time])
        month = np.asarray([t.month for t in time])

    # Store in a dataframe
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UnitStrippedWarning,
            message="The unit of the quantity is stripped",
        )
        df = pd.DataFrame(
            {"hemi": hemi, "year": year, "month": month, "track_id": track_id}
        )
    # Most frequent year, month and hemisphere for each track
    # Grouping is done to avoid labelling differently points in a track that might cross
    # hemisphere or seasons.
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
