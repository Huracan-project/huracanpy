import pytest
import numpy as np

import huracanpy
from huracanpy._data._netcdf import _find_trajectory_id


def test_load_track():
    data = huracanpy.load(huracanpy.example_TRACK_file, tracker="TRACK")
    assert len(data.groupby("track_id")) == 2


def test_load_csv():
    data = huracanpy.load(huracanpy.example_csv_file)
    assert len(data) == 13
    assert len(data.time) == 99
    assert len(data.groupby("track_id")) == 3


def test_load_netcdf():
    data = huracanpy.load(huracanpy.example_TRACK_netcdf_file)
    assert len(data.time) == 4580
    track_id = _find_trajectory_id(data)
    assert len(track_id) == 4580
    assert len(np.unique(track_id)) == 86


def test_load_tempest():
    data = huracanpy.load(huracanpy.example_TE_file, tracker="tempestextremes")

    assert len(data.time) == 210
    assert len(data.track_id) == 210
    assert len(data.groupby("track_id")) == 8


@pytest.mark.parametrize(
    "filename,tracker",
    [
        (huracanpy.example_TRACK_file, "TRACK"),
        (huracanpy.example_TRACK_netcdf_file, None),
        (huracanpy.example_csv_file, None),
        (huracanpy.example_TE_file, "tempestextremes"),
    ],
)
def test_save_netcdf(filename, tracker, tmp_path):
    data = huracanpy.load(filename, tracker=tracker)
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
            np.testing.assert_allclose(
                data[var].data.astype(data_[var].dtype), data_[var].data, rtol=1e-6
            )
        else:
            np.testing.assert_allclose(data[var].data, data_[var].data, rtol=0)


@pytest.mark.parametrize(
    "filename,tracker",
    [
        (huracanpy.example_TRACK_file, "TRACK"),
        (huracanpy.example_csv_file, None),
        (huracanpy.example_TE_file, "tempestextremes"),
    ],
)
def test_save_csv(filename, tracker, tmp_path):
    data = huracanpy.load(filename, tracker=tracker)
    # Copy the data because save modifies the dataset at the moment
    huracanpy.save(data.copy(), str(tmp_path / "tmp_file.csv"))

    # Reload the data and check it is still the same
    data_ = huracanpy.load(str(tmp_path / "tmp_file.csv"))

    for var in data_.variables:
        # Work around for xarray inconsistent loading the data as float or double
        # depending on fill_value and scale_factor
        # np.testing.assert_allclose doesn't work for datetime64
        if np.issubdtype(data[var].dtype, np.datetime64):
            assert (data[var].data == data_[var].data).all()
        elif data[var].dtype != data_[var].dtype:
            np.testing.assert_allclose(
                data[var].data.astype(data_[var].dtype), data_[var].data, rtol=1e-6
            )
        else:
            np.testing.assert_allclose(data[var].data, data_[var].data, rtol=0)


@pytest.mark.parametrize(
    "subset,length",
    [
        ("wmo", 8),
        ("usa", 10),
    ],
)
def test_ibtracs_offline(subset, length):
    ib = huracanpy.load(tracker="ibtracs", ibtracs_subset=subset)
    assert ib.season.min() == 1980
    assert (
        len(ib.time) > 0
    )  # Can't assert on dataset length, because it might change with updates.
    assert (len(ib)) == length
