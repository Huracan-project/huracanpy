"""
Module to load CHAZ tracks stored as NetCDF files.
"""

import xarray as xr
import numpy as np


def load(filename):
    raw_data = xr.open_dataset(filename)  # Read netcdf file
    stacked_data = raw_data.stack(
        record=[
            "stormID",
            "lifelength",
        ]
    )  # Stack stormID and lifelength into a record dimension
    return stacked_data.where(
        ~np.isnan(stacked_data.latitude), drop=True
    )  # Remove data that is only nans
