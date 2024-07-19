import pytest
import numpy as np

import huracanpy


def test_load_track():
    data = huracanpy.load(huracanpy.example_TRACK_file, tracker="TRACK")

    assert len(data) == 33
    assert len(data.coords) == 2
    assert len(data.time) == 46
    assert len(data.groupby("track_id")) == 2


def test_load_csv():
    data = huracanpy.load(huracanpy.example_csv_file)

    assert len(data) == 11
    assert len(data.coords) == 3
    assert len(data.time) == 99
    assert len(data.groupby("track_id")) == 3


def test_load_netcdf():
    data = huracanpy.load(huracanpy.example_TRACK_netcdf_file)

    assert len(data) == 19
    assert len(data.coords) == 18
    assert len(data.time) == 4580
    assert len(data.groupby("track_id")) == 86


def test_load_tempest():
    data = huracanpy.load(huracanpy.example_TE_file, tracker="tempestextremes")

    assert len(data) == 6
    assert len(data.coords) == 2
    assert len(data.time) == 210
    assert len(data.groupby("track_id")) == 8


def test_load_CHAZ():
    data = huracanpy.load(huracanpy.example_CHAZ_file, tracker="CHAZ")

    assert len(data.record) == 1078
    assert data.lifelength.max() == 124
    assert data.stormID.max() == 19


def test_load_MIT():
    data = huracanpy.load(huracanpy.example_MIT_file, tracker="MIT")

    assert len(data.record) == 3138
    assert data.time.max() == 1296000
    assert data.n_trk.max() == 10


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

    for var in list(data_.variables) + list(data_.coords):
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

    for var in list(data_.variables) + list(data_.coords):
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
        ("wmo", 6),
        ("usa", 8),
    ],
)
def test_ibtracs_offline(subset, length):
    ib = huracanpy.load(tracker="ibtracs", ibtracs_subset=subset)
    assert ib.season.min() == 1980
    assert (
        len(ib.time) > 0
    )  # Can't assert on dataset length, because it might change with updates.
    assert len(ib) == length
    assert len(ib.coords) == 3
