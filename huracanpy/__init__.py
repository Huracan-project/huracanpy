"""huracanpy - A python package for working with various forms of feature tracking data"""

__version__ = "0.1.0"
__author__ = "Leo Saffin <l.saffin@reading.ac.uk>, Stella Bourdin <stella.bourdin@physics.ox.ac.uk>, Kelvin Ng "
__all__ = [
    # Modules
    "diags",
    "plot",
    "assess",
    "utils",
    "tc",
    # Functions
    "load",
    "save",
    "sel_id",
    "trackswhere",
    # Parameters
    "basins",
    "example_csv_file",
    "example_parquet_file",
    "example_year_file",
    "example_TRACK_file",
    "example_TRACK_tilt_file",
    "example_TRACK_netcdf_file",
    "example_TE_file",
    "example_CHAZ_file",
    "example_MIT_file",
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
    example_TE_file,
    example_CHAZ_file,
    example_MIT_file,
)
from . import diags, plot, assess, utils, tc
from .utils._basins import basins_def as basins
from ._subset import sel_id, trackswhere
from . import _accessor
