"""
Module containing functions to compute rates.
"""

import warnings

import numpy as np
import xarray as xr
from metpy.units import units


def get_delta(var, track_ids=None, var_units=None, centering="forward"):
    """Take the differences across var, without including differences between the end
    and start of different tracks

    Parameters
    ----------
    var : xarray.DataArray
    track_ids : array_like, optional
    var_units : str, optional
    centering : str, optional

    Returns
    -------
    xarray.DataArray

    """
    # TODO: centered centering

    # Curate input
    # If track_id is not provided, all points are considered to belong to the same track
    if track_ids is None:
        track_ids = xr.DataArray([0] * len(var), dims=var.dims)
        warnings.warn(
            "track_id is not provided, all points are considered to come from the same"
            "track"
        )
    ## If time is provided, convert to numeric ns
    if var.dtype == "<M8[ns]":
        var = var.astype(float)
        var_units = "ns"
    ## Check that centering is supported
    if centering not in ["forward", "backward"]:
        raise ValueError("centering must be one of ['forward', 'backward']")

    # Compute delta
    delta = var[1:] - var[:-1]

    # Mask points where track_id changes
    tid_switch = track_ids[1:] == track_ids[:-1]
    delta = delta.where(tid_switch)

    # Apply centering
    if centering == "forward":
        delta = xr.concat([delta, xr.DataArray([np.nan], dims="record")], dim="record")
    elif centering == "backward":
        delta = xr.concat(
            [
                xr.DataArray([np.nan], dims="record"),
                delta,
            ],
            dim="record",
        )

    # return with units # TODO: If var has units, retrieve those
    if var_units is None:
        return xr.DataArray(delta, dims=var.dims)
    else:
        return xr.DataArray(delta, dims=var.dims) * units(var_units)


def get_rate(var, time, track_ids=None, var_units=None, centering="forward"):
    """Compute rate of change of var, without including differences between the end
    and start of different tracks

    Parameters
    ----------
    var : xarray.DataArray
    time : xarray.DataArray
    track_ids : array_like, optional
    var_units : str, optional
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
    dx = get_delta(var, track_ids, var_units=var_units, centering=centering)
    dt = get_delta(time, track_ids, centering=centering)
    dt = dt.metpy.convert_units("s")  # Convert to seconds

    return dx / dt
