"""
Module containing functions to compute rates.
"""

import warnings

import numpy as np
import xarray as xr
from metpy.units import units
from metpy.xarray import preprocess_and_wrap

from .._metpy import dequantify_results


@dequantify_results
@preprocess_and_wrap(wrap_like="var")
def delta(var, track_ids=None, centering="forward"):
    """Take the differences across var, without including differences between the end
    and start of different tracks

    Parameters
    ----------
    var : xarray.DataArray
    track_ids : array_like, optional
    centering : str, optional

    Returns
    -------
    xarray.DataArray

    """
    # TODO: centered centering

    # Curate input
    # If track_id is not provided, all points are considered to belong to the same track
    if track_ids is None:
        track_ids = np.zeros(var.shape)
        warnings.warn(
            "track_id is not provided, all points are considered to come from the same"
            "track"
        )

    # Check that centering is supported
    if centering not in ["forward", "backward"]:
        raise ValueError("centering must be one of ['forward', 'backward']")

    # Compute delta
    delta = var[1:] - var[:-1]

    # Mask points where track_id changes
    # Multiplying np.nan by an array element gives us the correct type of nan for both
    # np.timedelta and pint.Quantity
    delta[track_ids[1:] != track_ids[:-1]] = np.nan * delta[0]

    # Apply centering
    if centering == "forward":
        delta = np.concatenate([delta, [np.nan * delta[0]]])
    elif centering == "backward":
        delta = np.concatenate([[np.nan * delta[0]], delta])

    # Fix for timedeltas
    if np.issubdtype(delta.magnitude.dtype, np.timedelta64):
        delta = delta / np.timedelta64(1, "s")
        delta = delta.magnitude * units("s")

    return delta


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
    centering : str, optional

    Returns
    -------
    xarray.DataArray

    """
    # Curate input
    # If track_id is not provided, all points are considered to belong to the same track
    if track_ids is None:
        track_ids = xr.DataArray([0] * len(time), dims=time.dims)
        warnings.warn(
            "track_id is not provided, all points are considered to come from the same"
            "track"
        )
    ## Sort data by track_id and time
    # rate_var, track_ids, time = [a.sortby(time) for a in [rate_var, track_ids, time]]
    # rate_var, time, track_ids = [a.sortby(track_ids) for a in [rate_var, time, track_ids]]

    # TODO: If var has units, retrieve those

    # Compute deltas
    dx = delta(var, track_ids, centering=centering)
    dt = delta(time, track_ids, centering=centering)

    return dx / dt
