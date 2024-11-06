"""Module with function to interpolate the tracks"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr


def interp_time(
    data,
    freq="1h",
    track_id_name="track_id",
    prog_bar=False,
):
    """
    Function to interpolate track data at a given frequency.

    Parameters
    ----------
    data : xr.Dataset
        The tracks dataset.
    track_id_name : str, optional
        name of the track_id variable in data. The default is "track_id".
    freq : str, optional
        Frequency at which you want to interpolate the data. The default is '1h'.
    prog_bar : bool, optional

    Returns
    -------
    xarray.Dataset
        The input `xarray.Dataset` with each individual track interpolated to the
        requested frequencu
    """

    T = []

    if prog_bar:
        iter = tqdm(np.unique(data[track_id_name].values))
    else:
        iter = np.unique(data[track_id_name].values)

    for tid in iter:
        t = data.where(data[track_id_name] == tid, drop=True)
        T.append(
            t.set_coords("time")
            .swap_dims({"record": "time"})
            .interp(
                time=pd.date_range(t.time.min().values, t.time.max().values, freq=freq)
            )
            .swap_dims({"time": "record"})
        )
    return xr.concat(T, dim="record")
