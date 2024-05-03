"""
Module to load tracks stored as csv files, including TempestExtremes output.
"""

import numpy as np
import xarray as xr
import pandas as pd

def get_time(year, month, day, hour):
    """
    Get np.datetime64 array corresponding to year, month, day and hour arrays

    Parameters
    ----------
    year (np.array or pd.Series)
    month (np.array or pd.Series)
    day (np.array or pd.Series)
    hour (np.array or pd.Series)

    Returns
    -------
    np.array or pd.Series
        The corresponding np.datetime64
    """
    time = pd.to_datetime(
        year.astype(str)
        + "-"
        + month.astype(str)
        + "-"
        + day.astype(str)
        + " "
        + hour.astype(str)
        + ":00"
    )
    return time


def load(filename,):
    """Load csv tracks data as an xarray.Dataset
    These tracks may come from TempestExtremes StitchNodes, or any other source.

    Parameters
    ----------
    filename : str
        The file must contain at least longitude, latitude, time and track ID. 
            - longitude and latitude can be named that, or lon and lat.
            - time must be defined a single `time`column or by four columns : year, month, day, hour 
            - track ID must be within a column named track_id.

    Returns
    -------
    xarray.Dataset
    """
    
    ## Read file
    tracks = pd.read_csv(filename)
    if tracks.columns.str[0][1] == " ": # Sometimes columns names are read starting with a space, which we remove
        tracks = tracks.rename(columns={c: c[1:] for c in tracks.columns[1:]})
    tracks = tracks.rename({"longitude":"lon", "latitude":"lat"}) # Rename lon & lat columns if necessary

    ## Geographical attributes
    tracks.loc[tracks.lon < 0, "lon"] += 360 # Longitude are converted to [0,360] if necessary
    # TODO : Move it (^) to the wrapper level ?
    #tracks["hemisphere"] = np.where(tracks.lat > 0, "N", "S")
    # TODO : Determine basin (wrapper level?)

    ## Time attribute
    if "time" not in tracks.columns :
        tracks["time"] = get_time(tracks.year, tracks.month, tracks.day, tracks.hour)
    else :
        tracks["time"] = pd.to_datetime(tracks.time)
    # TODO : Determine season (wrapper level?)
    
    # Output xr dataset
    return tracks.to_xarray().rename({"index":"obs"})


