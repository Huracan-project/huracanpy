"""
Module containing functions to compute rates.
"""

import warnings

import numpy as np
from metpy.units import units
from metpy.xarray import preprocess_and_wrap

from .._metpy import dequantify_results


def _dummy_track_id(var):
    warnings.warn(
        "track_id is not provided, all points are considered to come from the sametrack"
    )
    return np.zeros(var.shape)


@dequantify_results
@preprocess_and_wrap(wrap_like="var")
def delta(var, track_ids=None, centering="forward"):
    """Take the differences across var, without including differences between the end
    and start of different tracks

    Parameters
    ----------
    var : xarray.DataArray
    track_ids : array_like, optional
    centering: str, optional
        - "forward" gives the delta based on the track point and the following track
          point. The last point of each track will be NaN
        - "backward" gives the delta based on the track point and the previous track
          point. The first point of each track will be NaN
        - "centre" gives the delta based on the centred difference of track points. The
          first and last points of each track will be NaN
        - "adaptive" gives the same as centred, but fills in the first point of each
          track with the forward difference, and the last point of each track with the
          backward difference

    Returns
    -------
    xarray.DataArray

    """
    # Curate input
    # If track_id is not provided, all points are considered to belong to the same track
    if track_ids is None:
        track_ids = _dummy_track_id(var)

    # Compute delta
    diff = var[1:] - var[:-1]

    # Apply centering
    diff = _align_array(diff, track_ids, centering)

    # Fix for timedeltas
    if np.issubdtype(diff.magnitude.dtype, np.timedelta64):
        diff = diff / np.timedelta64(1, "s")
        diff = diff.magnitude * units("s")

    return diff


@dequantify_results
@preprocess_and_wrap(wrap_like="var")
def rate(var, time, track_ids=None, centering="forward"):
    """Compute rate of change of var, without including differences between the end
    and start of different tracks

    Parameters
    ----------
    var : xarray.DataArray
    time : xarray.DataArray
    track_ids : array_like, optional
    centering: str, optional
        - "forward" gives the rate based on the track point and the following track
          point. The last point of each track will be NaN
        - "backward" gives the rate based on the track point and the previous track
          point. The first point of each track will be NaN
        - "centre" gives the rate based on the centred difference of track points. The
          first and last points of each track will be NaN
        - "adaptive" gives the same as centred, but fills in the first point of each
          track with the forward difference, and the last point of each track with the
          backward difference

    Returns
    -------
    xarray.DataArray

    """
    # Curate input
    # If track_id is not provided, all points are considered to belong to the same track
    if track_ids is None:
        track_ids = _dummy_track_id(var)

    ## Sort data by track_id and time
    # rate_var, track_ids, time = [a.sortby(time) for a in [rate_var, track_ids, time]]
    # rate_var, time, track_ids = [
    #     a.sortby(track_ids) for a in [rate_var, time, track_ids]
    # ]

    # Compute deltas
    dx = delta(var, track_ids, centering=centering)
    dt = delta(time, track_ids, centering=centering)

    return dx / dt


def _align_array(array, track_id, centering):
    # Index n, where the track ID changes between n and n+1
    # array is already a difference. So index n in array
    transition_points = np.where(track_id[1:] != track_id[:-1])[0]

    # Mask points where track_id changes
    # Multiplying np.nan by an array element gives us the correct type of nan for both
    # np.timedelta and pint.Quantity
    array[transition_points] = np.nan * array[0]

    if centering in ["centre", "center", "adaptive"]:
        centred = 0.5 * (array[1:] + array[:-1])
        centred = np.concatenate([[np.nan * array[0]], centred, [np.nan * array[0]]])

        if centering in ["centre", "center"]:
            return centred
        else:
            # Replace start/end points with forward and backward deltas
            centred[0] = array[0]
            centred[transition_points] = array[transition_points - 1]
            centred[transition_points + 1] = array[transition_points + 1]
            centred[-1] = array[-1]

            return centred

    elif centering == "forward":
        return np.concatenate([array, [np.nan * array[0]]])

    elif centering == "backward":
        return np.concatenate([[np.nan * array[0]], array])

    else:
        raise ValueError(
            f"Option align='{centering}' not recognised. Use one of"
            f" ['forward', 'backward', 'centre'/'center', 'adaptive']"
        )
