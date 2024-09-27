import numpy.exceptions
import pytest
import numpy as np

import huracanpy


@pytest.mark.parametrize(
    "filename, kwargs, nvars, ncoords, npoints, ntracks",
    [
        (huracanpy.example_TRACK_file, dict(tracker="TRACK"), 35, 0, 46, 2),
        (huracanpy.example_TRACK_tilt_file, dict(tracker="TRACK.tilt"), 3, 1, 46, 2),
        (huracanpy.example_csv_file, dict(), 13, 0, 99, 3),
        (huracanpy.example_parquet_file, dict(), 13, 0, 99, 3),
        (huracanpy.example_TRACK_netcdf_file, dict(), 20, 17, 4580, 86),
        (huracanpy.example_TE_file, dict(tracker="tempestextremes"), 8, 0, 210, 8),
        # (huracanpy.example_CHAZ_file, dict(tracker="CHAZ"), 9, 3, 1078, 20),
        # (huracanpy.example_MIT_file, dict(tracker="MIT"), 8, 4, 2138, 20),
        (None, dict(tracker="ibtracs", ibtracs_subset="wmo"), 8, 0, 139416, 4380),
        (None, dict(tracker="ibtracs", ibtracs_subset="usa"), 10, 0, 118882, 4072),
    ],
)
def test_load(filename, kwargs, nvars, ncoords, npoints, ntracks):
    data = huracanpy.load(filename, **kwargs)

    assert len(data) == nvars
    assert len(data.coords) == ncoords
    assert len(data.time) == npoints
    assert len(data.groupby("track_id")) == ntracks
    assert "record" not in data.coords


def test_load_CHAZ():
    data = huracanpy.load(huracanpy.example_CHAZ_file, tracker="CHAZ")

    assert len(data.record) == 1078
    assert data.lifelength.max() == 124
    assert data.stormID.max() == 19


def test_load_MIT():
    data = huracanpy.load(huracanpy.example_MIT_file, tracker="MIT")

    assert len(data.record) == 2138
    assert data.time.max() == 1119600
    assert data.n_track.max() == 10


@pytest.mark.parametrize(
    "filename, tracker",
    [
        (huracanpy.example_TRACK_file, "TRACK"),
        (huracanpy.example_TRACK_tilt_file, "TRACK.tilt"),
        (huracanpy.example_TRACK_netcdf_file, None),
        (huracanpy.example_csv_file, None),
        (huracanpy.example_parquet_file, None),
        (huracanpy.example_TE_file, "tempestextremes"),
        # (huracanpy.example_CHAZ_file, "CHAZ"),
        # (huracanpy.example_MIT_file, "MIT"),
        (None, "ibtracs"),
    ],
)
@pytest.mark.parametrize("extension", ["csv", "nc"])
@pytest.mark.parametrize("muddle", [False, True])
def test_save(filename, tracker, extension, muddle, tmp_path):
    if extension == "csv" and (
        filename == huracanpy.example_TRACK_tilt_file
        or filename == huracanpy.example_TRACK_netcdf_file
    ):
        pytest.skip(
            "The netCDF file has multiple dimensions so fails because converting to a"
            " dataframe leads to having rows equal to the product of the dimensions"
            " even though the dimensions cover different variables"
        )
    data = huracanpy.load(filename, tracker=tracker)

    # Check that save/load gives the same result when the track_id is not monotonic
    # Caused an issue because they got sorted before
    if muddle:
        data = data.sortby("track_id", ascending=False)

    # Copy the data because save modifies the dataset at the moment
    huracanpy.save(data.copy(), str(tmp_path / f"tmp_file.{extension}"))

    # Reload the data and check it is still the same
    data_ = huracanpy.load(str(tmp_path / f"tmp_file.{extension}"))

    assert len(data.variables) == len(data_.variables)
    assert len(data.coords) == len(data_.coords)
    for var in list(data.variables) + list(data.coords):
        # Work around for xarray inconsistent loading the data as float or double
        # depending on fill_value and scale_factor
        # np.testing.assert_allclose doesn't work for datetime64, object, or string
        if np.issubdtype(data[var].dtype, np.number):
            if data[var].dtype != data_[var].dtype:
                rtol = 1e-6
            else:
                rtol = 0
            np.testing.assert_allclose(
                data[var].data.astype(data_[var].dtype), data_[var].data, rtol=rtol
            )
        else:
            assert (data[var].data == data_[var].data).all()

    assert len(data.attrs) == len(data_.attrs)
    for attr in data.attrs:
        assert data.attrs[attr] == data_.attrs[attr]
