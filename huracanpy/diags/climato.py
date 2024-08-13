import pandas as pd
import xarray as xr


def nunique(self):
    return pd.Series(self).nunique()


xr.DataArray.nunique = nunique


def freq(self, by=None, track_id_name="track_id"):
    """
    Compute the frequency (number of tracks) in a dataset, potentially normalized by another variable (e.g. season to get yearly count).

    If you want to keep the season dimension to get for example the frequency time series, use groupby and apply:
    >>> data.groupby("season").apply(freq)
    You can also use groupby and apply to get for example yearly frequency per basin
    >>> data.groupby("basin").apply(freq, by = "season")

    Parameters
    ----------
    by : str, optional
        Variable to normalize frequency. The default is None (which means the number of unique track in the whole dataset is provided).
    track_id_name : str, optional
        Name of the unique track identifier variable. The default is "track_id".

    Returns
    -------
    xr.DataArray
        The frequency number.

    """
    if by is None:
        return xr.DataArray(self[track_id_name].nunique())
    else:
        return xr.DataArray(self[track_id_name].nunique() / self[by].nunique())
