"""Overlap function"""

from ._match import match


def overlap(tracks1, tracks2, matches=None):
    """
    Function computing the overlap between matched tracks.

    Parameters
    ----------
    tracks1 (xarray.Dataset)
    tracks2 (xarray.Dataset)
    matches (pandas.Dataframe): The output from match_tracks on tracks1 and tracks2.
        If None, match_tracks is run on tracks1 and tracks2.

    Returns
    -------
    pd.Dataframe
        Match dataset with added deltas in days
    """
    if matches is None:
        matches = match([tracks1, tracks2])
    c1, c2 = matches.columns[:2].str.slice(3)
    tracks1, tracks2 = tracks1.to_dataframe(), tracks2.to_dataframe()
    matches = (
        matches.join(
            tracks1.groupby("track_id")[["time"]]
            .min()
            .rename(columns={"time": "tmin_" + c1}),
            on="id_" + c1,
        )
        .join(
            tracks1.groupby("track_id")[["time"]]
            .max()
            .rename(columns={"time": "tmax_" + c1}),
            on="id_" + c1,
        )
        .join(
            tracks2.groupby("track_id")[["time"]]
            .min()
            .rename(columns={"time": "tmin_" + c2}),
            on="id_" + c2,
        )
        .join(
            tracks2.groupby("track_id")[["time"]]
            .max()
            .rename(columns={"time": "tmax_" + c2}),
            on="id_" + c2,
        )
    )

    matches["delta_start"] = matches["tmin_" + c2] - matches["tmin_" + c1]
    matches["delta_end"] = matches["tmax_" + c2] - matches["tmax_" + c1]
    matches["delta_end"] = (
        matches.delta_end.dt.days + matches.delta_end.dt.seconds / 86400
    )
    matches["delta_start"] = (
        matches.delta_start.dt.days + matches.delta_start.dt.seconds / 86400
    )

    return matches[["id_" + c1, "id_" + c2, "temp", "dist", "delta_start", "delta_end"]]
