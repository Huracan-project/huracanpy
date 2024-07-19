import pytest
import numpy as np

import huracanpy


@pytest.mark.parametrize(
    "filename, kwargs, nvars, ncoords, npoints, ntracks",
    [
        (huracanpy.example_TRACK_file, dict(tracker="TRACK"), 35, 0, 46, 2),
        (huracanpy.example_csv_file, dict(), 13, 1, 99, 3),
        (huracanpy.example_parquet_file, dict(), 13, 1, 99, 3),
        (huracanpy.example_TRACK_netcdf_file, dict(), 20, 17, 4580, 86),
        (huracanpy.example_TE_file, dict(tracker="tempestextremes"), 8, 0, 210, 8),
    ],
)
def test_load(filename, kwargs, nvars, ncoords, npoints, ntracks):
    data = huracanpy.load(filename, **kwargs)

    assert len(data) == nvars
    assert len(data.coords) == ncoords
    assert len(data.time) == npoints
    assert len(data.groupby("track_id")) == ntracks


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
    "filename, tracker",
    [
        (huracanpy.example_TRACK_file, "TRACK"),
        (huracanpy.example_TRACK_netcdf_file, None),
        (huracanpy.example_csv_file, None),
        (huracanpy.example_parquet_file, None),
        (huracanpy.example_TE_file, "tempestextremes"),
    ],
)
@pytest.mark.parametrize("extension", ["csv", "nc"])
def test_save(filename, tracker, extension, tmp_path):
    if filename == huracanpy.example_TRACK_netcdf_file and extension == "csv":
        pytest.skip(
            "The netCDF file has multiple dimensions so fails because converting to a"
            " dataframe leads to having rows equal to the product of the dimensions"
            " even though the dimensions cover different variables"
        )
    data = huracanpy.load(filename, tracker=tracker)
    # Copy the data because save modifies the dataset at the moment
    huracanpy.save(data.copy(), str(tmp_path / f"tmp_file.{extension}"))

    # Reload the data and check it is still the same
    data_ = huracanpy.load(str(tmp_path / f"tmp_file.{extension}"))

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
    assert len(ib) == length
    assert len(ib.coords) == 1
