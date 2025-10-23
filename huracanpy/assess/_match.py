"""Matching functions"""

from itertools import combinations, groupby

import numpy as np
import pandas as pd

from ..info import timestep
from ..calc import distance


def match(
    tracksets,
    names=None,
    max_dist=300,
    min_overlap=0,
    consecutive_overlap=False,
    tracks1_is_ref=False,
    distance_method="haversine",
):
    """
    Match the provided track sets between them.

    Parameters
    ----------
    tracksets : list[xarray.Dataset]
        list of track datasets to match together. Must be of length two or more.
    names : list, optional
        list of track datasets names. Must be the same size as tracksets. The default is
        ['1','2', ...].
    max_dist : float, optional
        Threshold for maximum distance between two tracks. The default is 300.
    min_overlap : int, optional
        Minimum number of overlapping time steps for matching. The default is 0.
    consecutive_overlap: bool, optional
        If min_overlap > 1, require that min_overlap points also need to be consective
    tracks1_is_ref: bool, optional
    distance_method: str, optional
        The method to use to calculate distance between track points.
        One of "haversine", "geod"

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the matching tracks with
            the id from corresponding datasets
            the number of matching time steps (if only two datasets provided)
            the distance between two tracks (if only two datasets provided)

    """

    # Check input
    if len(tracksets) < 2:
        raise ValueError("You must provide at least two datasets to match")
    if names is None:
        names = [str(n) for n in range(1, len(tracksets) + 1)]
    if len(names) != len(tracksets):
        raise ValueError(
            "Number of names provided do not correspond to the number of track sets"
        )

    # Two track sets
    if len(tracksets) == 2:
        return _match_pair(
            *tracksets,
            *names,
            max_dist,
            min_overlap,
            consecutive_overlap,
            tracks1_is_ref,
            distance_method,
        )
    # More than two track sets
    else:
        return _match_multiple(
            tracksets,
            names,
            max_dist,
            min_overlap,
            consecutive_overlap,
            tracks1_is_ref,
            distance_method,
        )


def _match_pair(
    tracks1,
    tracks2,
    name1="1",
    name2="2",
    max_dist=300,
    min_overlap=0,
    consecutive_overlap=False,
    tracks1_is_ref=False,
    distance_method="haversine",
):
    """

    Parameters
    ----------
    tracks1 (pd.DataFrame): the first track set to match
    tracks2 (pd.DataFrame): the second tracks set to match
    name1 (str): Suffix for the first dataframe
    name2 (str): Suffix for the second dataframe
    max_dist (float) : Threshold for maximum distance between two tracks
    min_overlap (int) : Minimum number of overlapping time steps for matching
    tracks1_is_ref (bool):
    distance_method (str): The method to use to calculate distance between track points.
        One of "haversine", "geod"

    Returns
    -------
    pd.DataFrame
        Dataframe containing the matching tracks with
            the id from both datasets
            the number of matching time steps
            the distance between two tracks
    """

    # Prepare data
    tracks1, tracks2 = (
        tracks1[["track_id", "lon", "lat", "time"]].to_dataframe(),
        tracks2[["track_id", "lon", "lat", "time"]].to_dataframe(),
    )
    tracks1["lon"] = np.where(tracks1.lon > 180, tracks1.lon - 360, tracks1.lon)
    tracks2["lon"] = np.where(tracks2.lon > 180, tracks2.lon - 360, tracks2.lon)

    # Find corresponding points (same time step, less than max_dist km)
    merged = pd.merge(tracks1, tracks2, on="time")

    if len(merged) == 0:
        # if there exist no matching points, return empty dataframe
        return pd.DataFrame(columns=["id_" + name1, "id_" + name2, "temp", "dist"])

    # if there exist matching points, continue
    # Calculate the distance between all points paired in time and subset points
    # that are within the threshold distance specified
    merged["dist"] = (
        distance(
            merged.lon_x,
            merged.lat_x,
            merged.lon_y,
            merged.lat_y,
            method=distance_method,
        )
        * 1e-3
    )
    merged = merged[merged.dist <= max_dist]

    # Precompute groupby
    track_groups = merged.groupby(["track_id_x", "track_id_y"])

    # Compute temporal overlap
    temp = track_groups[["dist"]].count().rename(columns={"dist": "temp"})

    if min_overlap >= 2:
        if consecutive_overlap:
            # Replace temporal overlap with longest consecutive set of merged points
            # for each track
            dt = timestep(tracks1.time, tracks1.track_id)

            for (track_id_x, track_id_y), track in track_groups:
                if len(track) >= min_overlap:
                    nconsecutive = (
                        max(
                            [
                                sum([1 for _ in grouper]) if value else 0
                                for value, grouper in groupby(np.diff(track.time) == dt)
                            ]
                        )
                        + 1
                    )

                    temp.loc[(track_id_x, track_id_y)].temp = nconsecutive

        # Subset by tracks sharing min_overlap points
        temp = temp[temp.temp >= min_overlap]

    # if there exist no matching points, return empty dataframe
    if len(temp) == 0:
        return pd.DataFrame(columns=["id_" + name1, "id_" + name2, "temp", "dist"])

    # Build a table of all pairs of tracks still present in temp
    matches = (
        merged[["track_id_x", "track_id_y"]]
        .drop_duplicates()
        .join(temp, on=["track_id_x", "track_id_y"])
    ).dropna()

    # Add average distance to the output
    dist = track_groups[["dist"]].mean()
    matches = matches.merge(dist, on=["track_id_x", "track_id_y"])

    # Treat duplicates if required
    if tracks1_is_ref:
        # Treat the duplicates where one tracks2 track has several corresponding
        # tracks1:
        # Keep the couple with the longest overlap
        matches = (
            matches.sort_values("temp", ascending=False)
            .groupby("track_id_y")
            .first()
            .reset_index()
        )

    # Rename columns before output
    matches = matches.rename(
        columns={"track_id_x": "id_" + name1, "track_id_y": "id_" + name2}
    )
    return matches


def _match_multiple(
    datasets,
    names,
    max_dist=300,
    min_overlap=0,
    consecutive_overlap=False,
    tracks1_is_ref=False,
    distance_method="haversine",
):
    """
    Function to match any number of tracks sets

    Parameters
    ----------
    datasets : list of xr.Dataset
        list of the sets to be matched.
    names : list of str
        labels for the datasets. names must have the same length as datasets
    max_dist : float
        Threshold for maximum distance between two tracks
    min_overlap : int
        Minimum number of overlapping time steps for matching

    Raises
    ------
    NotImplementedError
        If two datasets have no match.

    Returns
    -------
    M : pd.dataframe
        table of matching tracks among all the datasets

    """
    matches = pd.DataFrame(columns=["id_" + n for n in names[:2]])
    for names_pair, dataset_pair in zip(
        combinations(names, 2), combinations(datasets, 2)
    ):
        m = _match_pair(
            *dataset_pair,
            *names_pair,
            max_dist,
            min_overlap,
            consecutive_overlap,
            tracks1_is_ref=tracks1_is_ref * (names_pair[0] == names[0]),
            distance_method=distance_method,
        )
        if len(m) == 0:
            raise NotImplementedError(
                "For the moment, the case where two datasets have no match is not"
                "handled. Problem raised by datasets " + str(names_pair)  # TODO
            )
        matches = matches.merge(
            m[["id_" + names_pair[0], "id_" + names_pair[1]]], how="outer"
        )
    return matches


# TODO: Deal with duplicates: merge, max...?
