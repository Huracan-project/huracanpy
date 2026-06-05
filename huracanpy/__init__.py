"""
huracanpy - A python package for working with various forms of feature tracking data
"""

__all__ = [
    # Modules
    "convert",
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
    "_test_ibtracs_netcdf_file",
    "_test_non_ragged_netcdf_file",
    "_accessor",
]

from . import _accessor, assess, calc, convert, info, plot, tc
# Heavy optional dependencies (geopandas, cartopy, metpy, matplotlib, seaborn) are
# imported lazily inside the functions that use them, so that `import huracanpy` itself
# remains fast even when those libraries are installed.
from ._basins import basins
from ._concat import concat_tracks
from ._data import (
    _test_ibtracs_netcdf_file,
    _test_non_ragged_netcdf_file,
    example_CHAZ_file,
    example_csv_file,
    example_ERA20C_file,
    example_IRIS_file,
    example_MIT_file,
    example_old_HURDAT_file,
    example_parquet_file,
    example_STORM_file,
    example_TE_file,
    example_TRACK_file,
    example_TRACK_netcdf_file,
    example_TRACK_tilt_file,
    example_TRACK_timestep_file,
    example_WiTRACK_file,
    example_year_file,
    load,
    save,
)
from ._interp import interp_time
from ._subset import sel_id, trackswhere
