import numpy as np
import xarray as xr

__all__ = ["trackswhere", "sel_id"]


def sel_id(tracks, track_ids, track_id):
    """Select a track or tracks from a dataset of tracks by ID

    Parameters
    ----------
    tracks : xarray.Dataset
    track_ids : xarray.DataArray
        The track_ids corresponding to the tracks Dataset
    track_id : Any
        The track ID or IDs to match in track_ids.

    Returns
    -------
    xarray.Dataset

    """
    if track_ids.ndim != 1:
        raise ValueError("track_ids must be 1d")

    if np.isscalar(track_id):
        track_id = [track_id]

    dim = track_ids.dims[0]
    idx = np.where(np.isin(track_ids, track_id))[0]

    return tracks.isel(**{dim: idx})


def trackswhere(tracks, track_ids, condition):
    """Subset tracks that verify a condition.

    e.g select all tracks that are solely in the Northern hemisphere:

    >>> tracks_subset = huracanpy.trackswhere(tracks, tracks.track_id, lambda x: (x.lat > 0).all())

    e.g. select all tracks that are category 2 at least once in their lifetime:

    >>> track_subset = huracanpy.trackswhere(
        tracks, tracks.track_id, lambda track: track.pressure_category.max() >= 2
        )

    Parameters
    ----------
    tracks : xarray.Dataset
    track_ids : xarray.DataArray
        The track_ids corresponding to the tracks Dataset
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
