"""
Module containing functions to compute rates.
"""

import numpy as np
import xarray as xr
from metpy.units import units


def get_delta(delta_var, track_ids=None, var_units=None, centering="forward"):
    # TODO: centered centering

    # Curate input
    ## If track_id is not provided, all points are considered to belong to the same track
    if track_ids is None:
        track_ids = xr.DataArray([0] * len(delta_var), dims=delta_var.dims)
        print(
            "track_id is not provided, all points are considered to come from the same track"
        )
    ## If time is provided, convert to numeric ns
    if delta_var.dtype == "<M8[ns]":
        delta_var = delta_var.astype(float)
        var_units = "ns"
    ## Check that centering is supported
    assert centering in [
        "forward",
        "backward",
    ], "centering must be one of ['forward', 'backward']"

    # Compute delta
    delta = delta_var[1:] - delta_var[:-1]

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

    # return with units
    if var_units is None:
        return xr.DataArray(delta, dims=delta_var.dims)
    else:
        return xr.DataArray(delta, dims=delta_var.dims) * units(var_units)


def get_rate(rate_var, time, track_ids=None, var_units=None, centering="forward"):
    # Curate input
    ## If track_id is not provided, all points are considered to belong to the same track
    if track_ids is None:
        track_ids = xr.DataArray([0] * len(time), dims=time.dims)
        print(
            "track_id is not provided, all points are considered to come from the same track"
        )
    ## Sort data by track_id and time
    # rate_var, track_ids, time = [a.sortby(time) for a in [rate_var, track_ids, time]]
    # rate_var, time, track_ids = [a.sortby(track_ids) for a in [rate_var, time, track_ids]]

    # Compute deltas
    dx = get_delta(rate_var, track_ids, var_units=var_units, centering=centering)
    dt = get_delta(time, track_ids, centering=centering)
    dt = dt.metpy.convert_units("s")  # Convert to seconds

    return dx / dt
