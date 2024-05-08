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


@pytest.mark.parametrize("filename,tracker", [
    (huracanpy.example_TRACK_file, "TRACK"),
    (huracanpy.example_TRACK_netcdf_file, None),
    (huracanpy.example_csv_file, None),
])
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
            np.testing.assert_allclose(data[var].data.astype(data_[var].dtype), data_[var].data, rtol=1e-6)
        else:
            np.testing.assert_allclose(data[var].data, data_[var].data, rtol=0)

@pytest.mark.parametrize("filename,tracker", [
    (huracanpy.example_TRACK_file, "TRACK"),
    (huracanpy.example_TRACK_netcdf_file, None),
    (huracanpy.example_csv_file, None),
])
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
            np.testing.assert_allclose(data[var].data.astype(data_[var].dtype), data_[var].data, rtol=1e-6)
        else:
            np.testing.assert_allclose(data[var].data, data_[var].data, rtol=0)


def test_hemisphere():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    assert np.unique(huracanpy.utils.geography.get_hemisphere(data.lat)) == np.array(["S"])

def test_basin():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    assert huracanpy.utils.geography.get_basin(data.lon, data.lat)[0] == "AUS"
    assert huracanpy.utils.geography.get_basin(data.lon, data.lat)[-1] == "SI"

def test_sshs():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    assert huracanpy.utils.category.get_sshs_cat(data.wind10).min() == -1
    assert huracanpy.utils.category.get_sshs_cat(data.wind10).max() == 0

def test_pressure_cat():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    Klotz = huracanpy.utils.category.get_pressure_cat(data.slp/100)
    Simps = huracanpy.utils.category.get_pressure_cat(data.slp/100, convention = "Simpson")
    assert Klotz.sum() == 62
    assert Simps.sum() == -23
    
def test_seasons():
    data= huracanpy.load(huracanpy.example_year_file)
    season = huracanpy.utils.time.get_season(data.track_id, data.lat, data.time)
    assert season.str.len().min() == 4 
    assert season.str.len().max() == 8
    assert season.astype(int).min() == 1996
    assert season.astype(int).max() == 19961997
    np.testing.assert_approx_equal(season.astype(int).mean(), 7039001.37598945, 1)
    
def test_simple_track_density():
    data= huracanpy.load(huracanpy.example_year_file)
    D = huracanpy.diags.track_density.simple_global_histogram(data.lon, data.lat)
    assert D.min() == 1.
    assert D.max() == 43.
    assert D.median() == 4.
    assert np.isnan(D).sum() == 2240
    
def test_duration():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    d = huracanpy.diags.track_stats.duration(data)
    assert np.timedelta64(d.min().values, 'h') == np.timedelta64(126, 'h')
    assert np.timedelta64(d.max().values, 'h') == np.timedelta64(324, 'h')
    assert np.timedelta64(d.mean().values, 'h') == np.timedelta64(210, 'h')
    
def test_gen_vals():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    G = huracanpy.diags.track_stats.gen_vals(data)
    assert G.day.mean() == 10
    
def test_extremum_vals():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    M = huracanpy.diags.track_stats.extremum_vals(data, "wind10", "max")
    m = huracanpy.diags.track_stats.extremum_vals(data, "slp", "min")
    assert M.day.mean() == 15
    assert m.lat.mean() == -27