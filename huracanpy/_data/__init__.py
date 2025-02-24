"""huracanpy module for shared data"""

__all__ = [
    "load",
    "save",
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
]

import pathlib

from ._load import load
from ._save import save

here = pathlib.Path(__file__).parent
testdata_dir = here / "example_data"

example_TRACK_file = str(
    testdata_dir / "tr_trs_pos.2day_addT63vor_addmslp_add925wind_add10mwind.tcident.new"
)

example_TRACK_tilt_file = str(testdata_dir / "tilt.dat")

example_csv_file = str(testdata_dir / "sample.csv")

example_parquet_file = str(testdata_dir / "sample.parquet")

example_year_file = str(testdata_dir / "ERA5_1996_UZ.csv")

example_TRACK_netcdf_file = str(
    testdata_dir
    / "tr_trs_pos.2day_addT63vor_addmslp_add925wind_add10mwind.tcident.new.nc"
)

example_TRACK_timestep_file = str(
    testdata_dir
    / "tr_trs_pos.2day_addT63vor_addmslp_add925wind_add10mwind.nolat.tcident.hart_sample"
)
example_TE_file = str(testdata_dir / "TempestExtremes-sample.txt")

example_CHAZ_file = str(testdata_dir / "CHAZ_sample.nc")
example_MIT_file = str(testdata_dir / "MIT_sample.nc")
example_ERA20C_file = str(testdata_dir / "ERA20C_TRACK_NATL_tcident.nc")
example_WiTRACK_file = str(testdata_dir / "event_tracks_2017_ERA5.dat")

example_old_HURDAT_file = str(testdata_dir / "00012024050100_050_0005160_atl")
