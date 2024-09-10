"""
Module to load MIT (K. Emanuel and J. Lin's synthetic track generator) tracks stored as NetCDF files.
"""

import xarray as xr
import numpy as np


def load(filename, n_track_name="n_track", lat_track_name="lat_track"):
    raw_data = xr.open_dataset(filename)  # Read netcdf file
    stacked_data = raw_data.stack(
        record=[
            n_track_name,
            "time",
        ]
    )  # Stack n_trk and time into a record dimension
    return stacked_data.where(
        ~np.isnan(stacked_data[lat_track_name]), drop=True
    )  # Remove data that is only nans
