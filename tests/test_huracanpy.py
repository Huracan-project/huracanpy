import pytest
import numpy as np

import huracanpy


def test_load_track():
    data = huracanpy.load(huracanpy.example_TRACK_file, tracker="TRACK")
    assert len(data.groupby("track_id")) == 2


def test_load_csv():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="tempestextremes")
    assert len(data) == 13
    assert len(data.obs) == 99
    assert len(data.groupby("track_id")) == 3


def test_load_netcdf():
    data = huracanpy.load(huracanpy.example_TRACK_netcdf_file)
    assert len(data.time) == 4580
    track_id = huracanpy._tracker_specific.netcdf._find_trajectory_id(data)
    assert len(track_id) == 4580
    assert len(np.unique(track_id)) == 86


@pytest.mark.parametrize(
    "filename,tracker",
    [
        (huracanpy.example_TRACK_file, "TRACK"),
        (huracanpy.example_TRACK_netcdf_file, None),
        (huracanpy.example_csv_file, None),
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


def test_hemisphere():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    assert np.unique(huracanpy.utils.geography.get_hemisphere(data)) == np.array(["S"])


def test_basin():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    assert huracanpy.utils.geography.get_basin(data)[0] == "AUS"
    assert huracanpy.utils.geography.get_basin(data)[-1] == "SI"


def test_sshs():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    assert huracanpy.utils.category.get_sshs_cat(data.wind10).min() == -1
    assert huracanpy.utils.category.get_sshs_cat(data.wind10).max() == 0


def test_pressure_cat():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    Klotz = huracanpy.utils.category.get_pressure_cat(data.slp / 100)
    Simps = huracanpy.utils.category.get_pressure_cat(
        data.slp / 100, convention="Simpson"
    )
    assert Klotz.sum() == 62
    assert Simps.sum() == -23


def test_categorise():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    cat_orig = huracanpy.utils.category.get_sshs_cat(data.wind10)

    thresholds = huracanpy.utils.category._wind_thresholds["Saffir-Simpson"]
    cat_new = huracanpy.utils.category.categorise(data.wind10, thresholds)

    assert (cat_orig == cat_new).all()


def test_categorise_full():
    # Test with made up data for each category
    data = np.array([-1e24, 0, 20, 30, 40, 50, 60, 70, 1e24, np.nan])

    expected = np.array([-1, -1, 0, 1, 2, 3, 4, 5, 5, np.nan])
    result = huracanpy.utils.category.categorise(
        data, huracanpy.utils.category._wind_thresholds["Saffir-Simpson"]
    )

    # Separate test for last value (nan)
    assert (result[:-1] == expected[:-1]).all()
    assert np.isnan(result[-1])
