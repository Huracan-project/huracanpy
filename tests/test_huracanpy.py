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
        (huracanpy.example_TRACK_netcdf_file, dict(source="netcdf"), 20, 17, 4580, 86),
        (huracanpy.example_TRACK_timestep_file, dict(source="TRACK"), 38, 0, 416, 19),
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
        (
            huracanpy.example_TE_file,
            dict(source="tempestextremes", variable_names=["slp", "wind"]),
            8,
            0,
            210,
            8,
        ),
        (huracanpy.example_CHAZ_file, dict(), 11, 0, 1078, 20),
        (huracanpy.example_MIT_file, dict(), 10, 1, 2138, 11),
        (huracanpy.example_WiTRACK_file, dict(source="witrack"), 14, 0, 3194, 268),
        (None, dict(source="ibtracs", ibtracs_subset="wmo"), 8, 0, 143287, 4540),
        (None, dict(source="ibtracs", ibtracs_subset="usa"), 10, 0, 121806, 4170),
        (huracanpy.example_old_HURDAT_file, dict(source="ecmwf"), 8, 0, 183, 29),
    ],
)
def test_load(filename, kwargs, nvars, ncoords, npoints, ntracks):
    data = _load_with_checked_warnings(filename, **kwargs)

    assert len(data) == nvars
    assert len(data.coords) == ncoords
    assert len(data.time) == npoints
    assert len(data.groupby("track_id")) == ntracks
    assert "record" not in data.coords

    if filename != huracanpy.example_TRACK_tilt_file:
        for name in ["track_id", "time", "lon", "lat"]:
            assert name in data


def _fake_ibtracs_data(url, filename):  # noqa ARG001
    return huracanpy.example_csv_file, None


def test_load_ibtracs_online(monkeypatch):
    with monkeypatch.context() as m:
        from huracanpy._data import ibtracs

        m.setattr(ibtracs, "urlretrieve", _fake_ibtracs_data)
        tracks = huracanpy.load(source="ibtracs", ibtracs_subset="last3years")

    assert len(tracks) == 9
    assert len(tracks.coords) == 0
    # IBTrACS online load skips the second line (first line after header) so this is one
    # less than expected
    assert len(tracks.time) == 98
    assert len(tracks.groupby("track_id")) == 3
    assert "record" not in tracks.coords

    for name in ["track_id", "time", "lon", "lat"]:
        assert name in tracks


@pytest.mark.parametrize(
    "filename, kwargs, error, message",
    [
        ("", dict(source="nonsense"), ValueError, "Source nonsense unsupported"),
        ("", dict(), ValueError, "Source is set to None"),
    ],
)
def test_load_fails(filename, kwargs, error, message):
    with pytest.raises(error, match=message):
        huracanpy.load(filename, **kwargs)


def test_load_rename():
    tracks = huracanpy.load(
        huracanpy.example_csv_file,
        rename=dict(slp="pressure", not_a_variable="should_be_ignored"),
    )

    assert "pressure" in tracks
    assert "slp" not in tracks
    assert "should_be_ignored" not in tracks


def test_load_units():
    tracks = huracanpy.load(huracanpy.example_csv_file, units=dict(slp="Pa"))

    assert tracks.slp.attrs["units"] == "Pa"

    slp_hpa = tracks.slp.metpy.convert_units("hPa")

    np.testing.assert_allclose(tracks.slp.values, slp_hpa.data.magnitude * 100)


def test_load_baselon():
    tracks = huracanpy.load(huracanpy.example_csv_file, baselon=1000)

    assert tracks.lon.max() <= 1360
    assert tracks.lon.min() >= 1000


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
@pytest.mark.parametrize(
    "muddle, use_accessor", [(False, False), (True, False), (False, True)]
)
def test_save(filename, source, extension, muddle, use_accessor, tmp_path):
    if extension == "csv" and (
        filename == huracanpy.example_TRACK_tilt_file
        or (filename is not None and filename.split(".")[-1] == "nc")
    ):
        pytest.skip(
            "The netCDF file has multiple dimensions so fails because converting to a"
            " dataframe leads to having rows equal to the product of the dimensions"
            " even though the dimensions cover different variables"
        )
    data = _load_with_checked_warnings(filename, source=source)

    # Check that save/load gives the same result when the track_id is not monotonic
    # Caused an issue because they got sorted before
    if muddle:
        data = data.sortby("track_id", ascending=False)

    # Copy the data because save modifies the dataset at the moment
    data_orig = data.copy()

    filename = str(tmp_path / f"tmp_file.{extension}")
    if use_accessor:
        data.hrcn.save(filename)
    else:
        huracanpy.save(data, filename)

    # Check that the original data is not modified by the save function
    _assert_dataset_identical(data_orig, data)

    # Reload the data and check it is still the same
    # Saving as netcdf does force sorting by track_id so apply this
    if extension == "nc":
        data = data.sortby("track_id")
    data_reload = huracanpy.load(filename)
    _assert_dataset_identical(data, data_reload)


def test_save_fails(tracks_csv):
    with pytest.raises(NotImplementedError, match="File format not recognized"):
        huracanpy.save(tracks_csv, "filename.unsupported_extension")


def _load_with_checked_warnings(filename, **kwargs):
    if filename is None:
        if "ibtracs_subset" not in kwargs or kwargs["ibtracs_subset"] == "wmo":
            with (
                pytest.warns(
                    UserWarning,
                    match="This offline function loads a light version of IBTrACS",
                ),
                pytest.warns(
                    UserWarning, match="You are loading the IBTrACS-WMO subset"
                ),
            ):
                data = huracanpy.load(filename, **kwargs)
        else:
            with pytest.warns(
                UserWarning,
                match="This offline function loads a light version of IBTrACS",
            ):
                data = huracanpy.load(filename, **kwargs)
    else:
        data = huracanpy.load(filename, **kwargs)

    return data


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
