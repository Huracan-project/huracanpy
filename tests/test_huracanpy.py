import pytest
import numpy as np

import huracanpy


@pytest.mark.parametrize(
    "filename, kwargs, nvars, ncoords, npoints, ntracks",
    [
        (huracanpy.example_TRACK_file, dict(source="TRACK"), 35, 0, 46, 2),
        (huracanpy.example_TRACK_tilt_file, dict(source="TRACK.tilt"), 3, 1, 46, 2),
        (huracanpy.example_csv_file, dict(), 9, 0, 99, 3),
        (huracanpy.example_parquet_file, dict(), 9, 0, 99, 3),
        (huracanpy.example_TRACK_netcdf_file, dict(), 20, 17, 4580, 86),
        (
            huracanpy.example_TRACK_timestep_file,
            dict(source="TRACK", track_calendar=("1940-01-01", 6)),
            38,
            0,
            416,
            19,
        ),
        (
            huracanpy.example_TRACK_timestep_file,
            dict(
                source="TRACK",
                track_calendar=(
                    np.datetime64("1940-01-01"),
                    np.timedelta64(6 * 60 * 60, "s"),
                ),
            ),
            38,
            0,
            416,
            19,
        ),
        (huracanpy.example_TE_file, dict(source="tempestextremes"), 8, 0, 210, 8),
        (huracanpy.example_CHAZ_file, dict(), 11, 0, 1078, 20),
        (huracanpy.example_MIT_file, dict(), 10, 1, 2138, 11),
        (huracanpy.example_WiTRACK_file, dict(source="witrack"), 14, 0, 3194, 268),
        (None, dict(source="ibtracs", ibtracs_subset="wmo"), 8, 0, 143287, 4540),
        (None, dict(source="ibtracs", ibtracs_subset="usa"), 10, 0, 121806, 4170),
        (huracanpy.example_old_HURDAT_file, dict(source="ecmwf"), 8, 0, 183, 29),
    ],
)
def test_load(filename, kwargs, nvars, ncoords, npoints, ntracks):
    data = huracanpy.load(filename, **kwargs)

    assert len(data) == nvars
    assert len(data.coords) == ncoords
    assert len(data.time) == npoints
    assert len(data.groupby("track_id")) == ntracks
    assert "record" not in data.coords

    if filename != huracanpy.example_TRACK_tilt_file:
        for name in ["track_id", "time", "lon", "lat"]:
            assert name in data


@pytest.mark.parametrize(
    "filename, source",
    [
        (huracanpy.example_TRACK_file, "TRACK"),
        (huracanpy.example_TRACK_tilt_file, "TRACK.tilt"),
        (huracanpy.example_TRACK_netcdf_file, None),
        (huracanpy.example_csv_file, None),
        (huracanpy.example_parquet_file, None),
        (huracanpy.example_TE_file, "tempestextremes"),
        (huracanpy.example_CHAZ_file, None),
        (huracanpy.example_MIT_file, None),
        (huracanpy.example_WiTRACK_file, "witrack"),
        (None, "ibtracs"),
    ],
)
@pytest.mark.parametrize("extension", ["csv", "nc"])
@pytest.mark.parametrize("muddle", [False, True])
def test_save(filename, source, extension, muddle, tmp_path):
    if extension == "csv" and (
        filename == huracanpy.example_TRACK_tilt_file
        or (filename is not None and filename.split(".")[-1] == "nc")
    ):
        pytest.skip(
            "The netCDF file has multiple dimensions so fails because converting to a"
            " dataframe leads to having rows equal to the product of the dimensions"
            " even though the dimensions cover different variables"
        )
    data = huracanpy.load(filename, source=source)

    # Check that save/load gives the same result when the track_id is not monotonic
    # Caused an issue because they got sorted before
    if muddle:
        data = data.sortby("track_id", ascending=False)

    # Copy the data because save modifies the dataset at the moment
    data_orig = data.copy()
    huracanpy.save(data, str(tmp_path / f"tmp_file.{extension}"))

    # Check that the original data is not modified by the save function
    _assert_dataset_identical(data_orig, data)

    # Reload the data and check it is still the same
    # Saving as netcdf does force sorting by track_id so apply this
    if extension == "nc":
        data = data.sortby("track_id")
    data_reload = huracanpy.load(str(tmp_path / f"tmp_file.{extension}"))
    _assert_dataset_identical(data, data_reload)


def _assert_dataset_identical(ds1, ds2):
    assert len(ds1.variables) == len(ds2.variables)
    assert len(ds1.coords) == len(ds2.coords)
    for var in list(ds1.variables) + list(ds1.coords):
        # Work around for xarray inconsistent loading the data as float or double
        # depending on fill_value and scale_factor
        # np.testing.assert_allclose doesn't work for datetime64, object, or string
        if np.issubdtype(ds1[var].dtype, np.number):
            if ds1[var].dtype != ds2[var].dtype:
                rtol = 1e-6
            else:
                rtol = 0
            np.testing.assert_allclose(
                ds1[var].data.astype(ds2[var].dtype), ds2[var].data, rtol=rtol
            )
        else:
            assert (ds1[var].data == ds2[var].data).all()

    assert len(ds1.attrs) == len(ds2.attrs)
    for attr in ds1.attrs:
        assert ds1.attrs[attr] == ds2.attrs[attr]
