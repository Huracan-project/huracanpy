"""huracanpy - A python package for working with various forms of feature tracking data"""

__version__ = "0.1.0"
__author__ = "Leo Saffin <l.saffin@reading.ac.uk>, Stella Bourdin <stella.bourdin@physics.ox.ac.uk>, Kelvin Ng "
__all__ = [
    "load",
    "save",
    "utils",
    "diags",
    "plot",
    "assess",
    "subset",
    "example_csv_file",
    "example_year_file",
    "example_TRACK_file",
    "example_TRACK_netcdf_file",
    "example_TE_file",
]


from . import utils
from ._data import (
    load,
    save,
    example_csv_file,
    example_year_file,
    example_TRACK_file,
    example_TRACK_netcdf_file,
    example_TE_file,
)
from . import diags
from . import plot
from . import assess
from . import subset
