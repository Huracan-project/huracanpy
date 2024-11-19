import numpy as np
import xarray as xr

__all__ = ["trackswhere", "sel_id"]


def sel_id(tracks, track_ids, track_id):
    """Select an individual track from a set of tracks by ID

    Parameters
    ----------
    tracks : xarray.Dataset
    track_ids : xarray.DataArray
        The track_ids corresponding to the tracks Dataset
    track_id : Any
        The track ID to match in track_ids. Must be the same type as the track_ids.
        Usually `int` or `str`

    Returns
    -------
    xarray.Dataset

    """
    if track_ids.ndim != 1:
        raise ValueError("track_ids must be 1d")

    dim = track_ids.dims[0]
    idx = np.where(track_ids == track_id)[0]

    return tracks.isel(**{dim: idx})


def trackswhere(tracks, track_ids, condition):
    """Subset tracks from the input

    e.g select all tracks that are solely in the Northern hemisphere
    >>> tracks_subset = huracanpy.trackswhere(tracks, lambda x: (x.lat > 0).all())

    Parameters
    ----------
    tracks : xarray.Dataset
    track_ids : xarray.DataArray
    condition : callable
        A function that takes an `xarray.Dataset` of an individual track and returns
        True or False

    Returns
    -------
    xarray.Dataset
        A dataset with the subset of tracks from the input that match the given criteria

    """
    if track_ids.ndim != 1:
        raise ValueError("track_ids must be 1d")

    track_groups = [
        track for track_id, track in tracks.groupby(track_ids) if condition(track)
    ]

    return xr.concat(track_groups, dim=track_ids.dims[0])
