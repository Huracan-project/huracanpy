"""huracanpy module for shared data"""

__all__ = [
    "load",
    "save",
    "example_csv_file",
    "example_year_file",
    "example_TRACK_file",
    "example_TRACK_netcdf_file",
]

import pathlib

from ._load import load
from ._save import save

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
