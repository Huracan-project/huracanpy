"""Matching functions"""

import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit
from itertools import combinations


def match_pair(
    tracks1,
    tracks2,
    name1="1",
    name2="2",
    max_dist=300,
    min_overlap=0,
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
    X = np.concatenate([[merged.lat_x], [merged.lon_x]]).T
    Y = np.concatenate([[merged.lat_y], [merged.lon_y]]).T
    merged["dist"] = haversine_vector(X, Y, unit=Unit.KILOMETERS)
    merged = merged[merged.dist <= max_dist]
    # Compute temporal overlap
    temp = (
        merged.groupby(["track_id_x", "track_id_y"])[["dist"]]
        .count()
        .rename(columns={"dist": "temp"})
    )
    # Build a table of all pairs of tracks sharing at least one point
    matches = (
        merged[["track_id_x", "track_id_y"]]
        .drop_duplicates()
        .join(temp, on=["track_id_x", "track_id_y"])
    )
    matches = matches[matches.temp >= min_overlap]
    dist = merged.groupby(["track_id_x", "track_id_y"])[["dist"]].mean()
    matches = matches.merge(dist, on=["track_id_x", "track_id_y"])
    # Rename columns before output
    matches = matches.rename(
        columns={"track_id_x": "id_" + name1, "track_id_y": "id_" + name2}
    )
    return matches


def match_multiple(
    datasets,
    names,
    max_dist=300,
    min_overlap=0,
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

    assert len(datasets) == len(names), "datasets and names must have the same length."

    M = pd.DataFrame(columns=["id_" + n for n in names[:2]])
    for names_pair, dataset_pair in zip(
        combinations(names, 2), combinations(datasets, 2)
    ):
        m = match_pair(*dataset_pair, *names_pair, max_dist, min_overlap)
        if len(m) == 0:
            raise NotImplementedError(
                "For the moment, the case where two datasets have no match is not handled. Problem raised by datasets "  # TODO
                + str(names_pair)
            )
        M = M.merge(m[["id_" + names_pair[0], "id_" + names_pair[1]]], how="outer")
    return M


# TODO: Deal with duplicates: merge, max...?
