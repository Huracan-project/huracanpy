import pandas as pd
import xarray as xr

from ._track_stats import duration
from huracanpy.utils.ace import get_ace
from metpy.units import units


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
        return self.groupby(by).apply(freq, by=None).mean()


def tc_days(self, by=None, track_id_name="track_id", time_name="time"):
    """
    Function to compute the number of "TC days", or cumulated TC duration, potentially normalized by another variable (e.g. season to get yearly TCD).

    NB: Duration is computed over the whole lifecycle that is provided in the dataset.
    If you want the cumulated duration of e.g. only stages with wind above a given threshold, use where to preliminary filter the dataset:
    >>> TC_days(data.where(data.wind>64), by="season") # Cumulates the number of days with wind above 64 <units>
    >>> TC_days(data.where(data.stage == "tropical"), by="season") # Cumulates the number of days where the stage has been identified as tropical

    Parameters
    ----------
    by : str, optional
        Variable to normalize frequency. The default is None (which means the cumulated duration in the whole dataset is provided).
    track_id_name : str, optional
        Name of the unique track identifier variable. The default is "track_id".
    time_name : str, optional
        Name of the time variable. The default is "time".

    Returns
    -------
    xr.DataArray
        The number of TC days, or cumulated TC duration.

    """
    if by is None:
        return xr.DataArray(duration(self[time_name], self[track_id_name]).sum() / 24)
    else:
        return self.groupby(by).apply(tc_days, by=None).mean()


def ace(
    self, by=None, wind_name="wind", threshold=0 * units("knots"), wind_units="m s-1"
):
    """
    Function to aggregate ACE.


    Parameters
    ----------
    by : str, optional
        Variable to normalize frequency. The default is None (which means the cumulated duration in the whole dataset is provided).
    wind_name : str, optional
        Name of the variable with wind to compute ACE. The default is "wind".
    threshold : scalar, default=0 knots
        ACE is set to zero below this threshold wind speed. The default argument is in
        knots. To pass an argument with units, use :py:mod:`metpy.units`, otherwise any
        non-default argument will be assumed to have the units of "wind_units" which is
        "m s-1" by default.
    wind_units : str, default="m s-1"
        If the units of wind are not specified in the attributes then the function will
        assume it is in these units before converting to knots

    Returns
    -------
    xr.DataArray
        Aggregated ACE.

    """
    ace = get_ace(self[wind_name], threshold, wind_units)

    if by is None:
        return ace.sum()
    else:
        return ace.groupby(self[by]).sum().mean()
