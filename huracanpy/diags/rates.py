"""
Module containing functions to compute rates.
"""

import numpy as np
import xarray as xr


def rate(data, rate_var="slp"):
    """
    Function to compute the evolution of a variable over time (rate). In particular, works for intensification and deepening rates.

    Parameters
    ----------
    data : xarray.dataset of tracks
    rate_var : str, optional
        Variable for which the rate is to be computed (ex: "wind" for intensification rate, "slp" for deepening rate). The default is "slp".

    Returns
    -------
    xarray.Dataset
        Rate. Output is stored for points that correspond to the middle of two consecutive points in the initial dataset.
        The rate is in <unit>/h, where <unit> is the unit of rate_var.
    """

    data = data.sortby(["track_id", "time"])

    # Compute rate
    rate = (data[rate_var].values[1:] - data[rate_var].values[:-1]) / (
        data.time.values[1:] - data.time.values[:-1]
    ).astype("timedelta64[h]").astype(int)

    # Output nice dataset
    t = data.time.values[:-1] + (data.time.values[1:] - data.time.values[:-1]) / 2
    lon = (data.lon.values[1:] + data.lon.values[:-1]) / 2
    lat = (data.lat.values[1:] + data.lat.values[:-1]) / 2
    mask = data.track_id.values[1:] == data.track_id.values[:-1]

    # Transform into clean dataset
    rate = xr.DataArray(
        rate, dims="mid_record", coords={"mid_record": np.arange(len(rate))}
    )
    lon = xr.DataArray(
        lon, dims="mid_record", coords={"mid_record": np.arange(len(lon))}
    )
    lat = xr.DataArray(
        lat, dims="mid_record", coords={"mid_record": np.arange(len(lat))}
    )
    t = xr.DataArray(t, dims="mid_record", coords={"mid_record": np.arange(len(t))})
    tid = xr.DataArray(
        data.track_id[1:],
        dims="mid_record",
        coords={"mid_record": np.arange(len(data.track_id[1:]))},
    )

    ds = xr.Dataset({"rate": rate, "lon": lon, "lat": lat, "time": t, "track_id": tid})

    # Remove values for transition between two tracks
    return ds.where(
        mask,
    )  # drop=True raises an error that I don't understand...
