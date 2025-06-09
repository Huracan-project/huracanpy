import numpy as np
import xarray as xr


def concat_tracks(
    objs, track_id="track_id", *, prefix=None, start=0, keep_track_id=False, **kwargs
):
    """Concatenate the tracks while making sure the track_id remains unique for each
    track

    Parameters
    ----------
    objs : sequence of xarray.Dataset
        The tracks to concatenate
    track_id : str, default="track_id"
        The name of the track_id variable in the tracks to concatenate
    prefix : str, optional
        A string to start each track_id to make them unique. Needs to be a
        python-formattable string e.g. `"{}_"` will result in track_id's that look like
        `{n}_{track_id}`, where `n` is the count of the DataArray in objs and `track_id`
        is the original track_id.
    start : int, optional
        The first entry for the new track IDs. If prefix is None, then the track_ids
        will have values from `start` to `start` + number_of_tracks
    keep_track_id : bool or str, default=False
        Save the original track_id as a new variable in the output Dataset. If `str`
        this gives the variable name, "track_id_original" otherwise
    **kwargs :
        Passed to :py:func:`xarray.concat`

    Returns
    -------

    """
    all_tracks = []

    current_track_id = start
    for n, tracks in enumerate(objs, start=start):
        track_id_old = tracks[track_id]

        if prefix is not None:
            prefix_ = xr.DataArray(prefix.format(n) + "{}")
            tracks = tracks.assign(
                {track_id: prefix_.str.format(track_id_old.astype(str))}
            )

        else:
            tracks = _reset_track_id(
                tracks,
                track_id_old,
                start=current_track_id,
                keep_original=keep_track_id,
            )
            current_track_id = tracks[track_id].values.max() + 1

        all_tracks.append(tracks)

    return xr.concat(all_tracks, dim=track_id_old.dims[0], **kwargs)


def _reset_track_id(tracks, track_ids, start=0, *, keep_original=False):
    """Replace the track IDs with an ascending sequence of numbers

    Examples
    >>> tracks1 = _reset_track_id(tracks1,tracks1.track_id)
    >>> tracks2 = _reset_track_id(
    >>>     tracks2,tracks2.track_id,start=tracks1.track_id.values.max() + 1
    >>> )
    >>> all_tracks = xr.concat([tracks1, tracks2], dim="record")

    Parameters
    ----------
    tracks : xarray.Dataset
    track_ids : array_like
        The current track IDs
    start : int, optional
        The first entry for the new track IDs. The track_ids will have values from
        `start` to `start` + number_of_tracks
    keep_original : bool or str, default=False
        Save the original track_id as a new variable in the output Dataset. If `str`
        this gives the variable name, "track_id_original" otherwise

    Returns
    -------
    xarray.Dataset:
        A copy of the input dataset with reassigned track_ids
    """
    if track_ids.ndim != 1:
        raise ValueError("track_ids must be 1d")

    _, new_track_ids = np.unique(track_ids, return_inverse=True)

    # Default name for original track id if not specified
    if keep_original is True:
        keep_original = "track_id_original"

    if isinstance(keep_original, str):
        tracks = tracks.assign(**{keep_original: (track_ids.dims[0], track_ids.values)})

    tracks = tracks.assign(
        **{track_ids.name: (track_ids.dims[0], start + new_track_ids)}
    )

    return tracks
