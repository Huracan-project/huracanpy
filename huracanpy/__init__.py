"""
huracanpy - A python package for working with various forms of feature tracking data
"""

__all__ = [
    # Modules
    "calc",
    "plot",
    "assess",
    "info",
    "tc",
    # Functions
    "load",
    "save",
    "sel_id",
    "trackswhere",
    "interp_time",
    "concat_tracks",
    # Parameters
    "basins",
    "example_csv_file",
    "example_parquet_file",
    "example_year_file",
    "example_TRACK_file",
    "example_TRACK_tilt_file",
    "example_TRACK_netcdf_file",
    "example_TRACK_timestep_file",
    "example_TE_file",
    "example_CHAZ_file",
    "example_MIT_file",
    "example_ERA20C_file",
    "example_WiTRACK_file",
    "example_old_HURDAT_file",
    "example_STORM_file",
    "example_IRIS_file",
    "_accessor",
]

from ._data import (
    load,
    save,
    example_csv_file,
    example_parquet_file,
    example_year_file,
    example_TRACK_file,
    example_TRACK_tilt_file,
    example_TRACK_netcdf_file,
    example_TRACK_timestep_file,
    example_TE_file,
    example_CHAZ_file,
    example_MIT_file,
    example_ERA20C_file,
    example_WiTRACK_file,
    example_old_HURDAT_file,
    example_STORM_file,
    example_IRIS_file,
)
from ._basins import basins
from ._interp import interp_time
from ._concat import concat_tracks
from ._subset import sel_id, trackswhere
from . import calc, plot, assess, info, tc

from . import _accessor
