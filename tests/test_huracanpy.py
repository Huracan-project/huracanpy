import pytest
import numpy as np

import huracanpy


def test_load_track():
    data = huracanpy.load(huracanpy.example_TRACK_file, tracker="TRACK")
    assert len(data.groupby("track_id")) == 2


def test_load_netcdf():
    data = huracanpy.load(huracanpy.example_TRACK_netcdf_file)
    assert len(data.time) == 4580
    track_id = huracanpy._find_trajectory_id(data)
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
        isnan = np.isnan(data[var].data)
        assert np.isnan(data_[var][isnan].data).all()
        assert (data[var].data[~isnan] == data_[var].data[~isnan]).all()
