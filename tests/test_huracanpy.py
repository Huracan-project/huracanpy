import pytest
import numpy as np

import huracanpy


def test_load_track():
    data = huracanpy.load(huracanpy.example_TRACK_file, tracker="TRACK")
    assert len(data.groupby("track_id")) == 2


def test_load_csv():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "tempestextremes")
    assert len(data) == 13
    assert len(data.obs) == 99
    assert len(data.groupby("track_id")) == 3


def test_load_netcdf():
    data = huracanpy.load(huracanpy.example_TRACK_netcdf_file)
    assert len(data.time) == 4580
    track_id = huracanpy._tracker_specific.netcdf._find_trajectory_id(data)
    assert len(track_id) == 4580
    assert len(np.unique(track_id)) == 86


@pytest.mark.parametrize("filename", [huracanpy.example_TRACK_file, huracanpy.example_TRACK_netcdf_file])
def test_save_netcdf(filename, tmp_path):
    data = huracanpy.load(filename, tracker="TRACK")
    # Copy the data because save modifies the dataset at the moment
    huracanpy.save(data.copy(), str(tmp_path / "tmp_file.nc"))

    # Reload the data and check it is still the same
    data_ = huracanpy.load(str(tmp_path / "tmp_file.nc"))

    for var in data_.variables:
        # Work around for xarray inconsistent loading the data as float or double
        # depending on fill_value and scale_factor
        # np.testing.assert_allclose doesn't work for datetime64
        if np.issubdtype(data[var].dtype, np.datetime64):
            assert (data[var].data == data_[var].data).all()
        elif data[var].dtype != data_[var].dtype:
            np.testing.assert_allclose(data[var].data.astype(data_[var].dtype), data_[var].data, rtol=1e-6)
        else:
            np.testing.assert_allclose(data[var].data, data_[var].data, rtol=0)
