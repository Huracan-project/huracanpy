"""
Utils related to storm category
"""

import numpy as np
import xarray as xr
import pandas as pd

def categorize(var, bins, labels=None):
    """
    Provides category according to provided bins and labels

    Parameters
    ----------
    var : xr.DataArray
        The variable to categorize
    bins : list or np.array
        bins boundaries
    labels : list or np.array, optional
        Name of the categories. len(labels) = len(bins) -1

    Returns
    -------
    None.

    """
    cat = pd.cut(var, bins, labels=labels)
    return xr.DataArray(cat, dims = "obs", coords = {"obs":var.obs})
    

def get_sshs_cat(wind): # TODO : Manage units properly (with pint?)
    """
    Function to determine the Saffir-Simpson Hurricane Scale (SSHS) category.

    Parameters
    ----------
    wind : xr.DataArray
        10-minutes averaged 10m wind in m/s

    Returns
    -------
    xarray.DataArray
        The category series. 
        You can append it to your tracks by running tracks["sshs"] = get_sshs_cat(tracks.wind)
    """
    
    bins_sshs = [0,16,29,38,44,52,63,np.inf]
    labels_sshs = [-1,0,1,2,3,4,5]
    
    return categorize(wind, bins_sshs, labels_sshs)

_slp_thresholds = {
    "Simpson" : np.flip([+np.inf, 990, 980, 970, 965, 945, 920, 0]),
    "Klotzbach" : np.flip([+np.inf, 1005, 990, 975, 960, 945, 925, 0])
    }

def get_pressure_cat(p, convention = "Klotzbach"):
    """
    Determine the pressure category according to selected convention.

    Parameters
    ----------
    slp : xr.DataArray
        Minimum Sea-level Pressure series in hPa
    convention : str
        Name of the classification convention you want to use. 
            * Klotzbach (default)
            * Simpson

    Returns
    -------
    xarray.DataArray
        The category series. 
        You can append it to your tracks by running tracks["cat"] = get_pressure_cat(tracks.slp)

    """
    
    return categorize(p, _slp_thresholds[convention], labels=np.flip(np.arange(-1,5+1)))