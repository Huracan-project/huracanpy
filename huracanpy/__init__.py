"""huracanpy - A python package for working with various forms of feature tracking data"""

__version__ = "0.1.0"
__author__ = "Leo Saffin <l.saffin@reading.ac.uk>, Stella Bourdin <stella.bourdin@physics.ox.ac.uk>, Kelvin Ng "
__all__ = ["load", "save", "utils", "diags", "plot", "data", "assess"]

import pathlib

from . import utils
from .load import load
from .save import save
from . import diags
from . import plot
from . import data
from . import assess


here = pathlib.Path(__file__).parent
testdata_dir = here / "example_data"

example_TRACK_file = str(
    testdata_dir / "tr_trs_pos.2day_addT63vor_addmslp_add925wind_add10mwind.tcident.new"
)

example_csv_file = str(testdata_dir / "sample.csv")

example_year_file = str(testdata_dir / "ERA5_1996_UZ.csv")

example_TRACK_netcdf_file = str(
    testdata_dir
    / "tr_trs_pos.2day_addT63vor_addmslp_add925wind_add10mwind.tcident.new.nc"
)
