"""Module with function to interpolate the tracks"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr

from ._subset import sel_id


def interp_time(
    tracks,
    track_ids,
    freq="1h",
    prog_bar=False,
):
    """
    Function to interpolate track data at a given frequency.

    Parameters
    ----------
    tracks : xarray.Dataset
        The tracks dataset.
    track_ids : xarray.DataArray
    freq : str, optional
        Frequency at which you want to interpolate the data. The default is '1h'.
    prog_bar : bool, optional

    Returns
    -------
    xarray.Dataset
        The input `xarray.Dataset` with each individual track interpolated to the
        requested frequency
    """

    T = []

    if prog_bar:
        iterator = tqdm(np.unique(track_ids.values))
    else:
        iterator = np.unique(track_ids.values)

    for tid in iterator:
        t = sel_id(tracks, track_ids, tid)
        T.append(
            t.set_coords("time")
            .swap_dims({"record": "time"})
            .interp(
                time=pd.date_range(t.time.min().values, t.time.max().values, freq=freq)
            )
            .swap_dims({"time": "record"})
            .reset_coords("time")
        )
    return xr.concat(T, dim="record")
