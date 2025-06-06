import pathlib

import pytest
import xarray as xr

import huracanpy


data_path = pathlib.Path(__file__).parent / "data_interpolate"


@pytest.mark.parametrize("prog_bar", [(True,), (False,)])
def test_interpolate_time(tracks_csv, prog_bar):
    result = huracanpy.interp_time(tracks_csv, tracks_csv.track_id, prog_bar=prog_bar)

    expected = huracanpy.load(str(data_path / "interpolate_time_result.nc"))

    # cf_role="trajectory_id" automatically added when result was saved
    del expected.track_id.attrs["cf_role"]

    xr.testing.assert_allclose(result, expected)
