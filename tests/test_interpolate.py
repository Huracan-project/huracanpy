import pathlib

import xarray as xr

import huracanpy


data_path = pathlib.Path(__file__).parent / "data_interpolate"


def test_interpolate_time(tracks_csv):
    result = huracanpy.interp_time(tracks_csv, tracks_csv.track_id)

    expected = huracanpy.load(str(data_path / "interpolate_time_result.nc"))

    # cf_role="trajectory_id" automatically added when result was saved
    del expected.track_id.attrs["cf_role"]

    xr.testing.assert_identical(result, expected)
