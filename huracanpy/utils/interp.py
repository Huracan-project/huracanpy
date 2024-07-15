"""Module with function to interpolate the tracks"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr


def interp_time(
    data,
    freq="1h",
    track_id_name="track_id",
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

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    T = []
    for tid in tqdm(np.unique(data[track_id_name].values)):
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
