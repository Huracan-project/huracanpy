import xarray as xr

__all__ = ["trackswhere", "sel_id"]


def sel_id(data, tid, track_id_name="track_id"):
    df = data.to_dataframe()
    track = df[df[track_id_name] == tid]
    return track.to_xarray()


def trackswhere(tracks, condition):
    """Subset tracks from the input

    e.g select all tracks that are solely in the Northern hemisphere
    >>> tracks_subset = huracanpy.trackswhere(tracks, lambda x: (x.lat > 0).all())

    Parameters
    ----------
    tracks : xarray.Dataset
    condition : function
        A function that takes an `xarray.Dataset` of an individual track and returns
        True or False

    Returns
    -------
    xarray.Dataset
        A dataset with the subset of tracks from the input that match the given criteria

    """
    track_groups = tracks.groupby("track_id")

    if callable(condition):
        is_match = track_groups.map(condition)

    track_groups = [
        track for n, (track_id, track) in enumerate(track_groups) if is_match[n]
    ]

    assert len(tracks.time.dims) == 1
    return xr.concat(track_groups, dim=tracks.time.dims[0])
