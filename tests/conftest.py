import pytest

import datetime
from collections import namedtuple

import cftime
import numpy as np
import xarray as xr

with pytest.warns(PendingDeprecationWarning, match="Overriding `Axes.cla` in GeoAxes"):
    import huracanpy


def pytest_addoption(parser):
    parser.addoption(
        "--docs",
        action="store_true",
        default=False,
        help="Test running documentation ipynb's",
    )


@pytest.fixture()
def tracks_csv():
    return huracanpy.load(huracanpy.example_csv_file)


@pytest.fixture()
def tracks_with_extra_coord(tracks_csv):
    # Test that the same results apply if a variable has an additional dimension to the
    # time/track_id dimension (e.g. if each point had a profile on pressure levels)
    # Most functions should work fine but using pandas can cause the data to be
    # broadcast across the dimensions to be able to represent it as 1d
    return tracks_csv.assign(
        thing=(
            (
                ("record", "level"),
                np.array(
                    [np.ones_like(tracks_csv.lon), np.ones_like(tracks_csv.lon) * 2],
                ).T,
            )
        )
    )


@pytest.fixture()
def tracks_year():
    return huracanpy.load(huracanpy.example_year_file)


@pytest.fixture()
def tracks_year_cftime():
    tracks = huracanpy.load(huracanpy.example_year_file)
    time = [
        cftime.datetime(t.dt.year, t.dt.month, t.dt.day, t.dt.hour) for t in tracks.time
    ]
    tracks["time"] = ("record", time)
    return tracks


@pytest.fixture()
def tracks_minus180_plus180():
    return xr.Dataset(
        dict(
            track_id=np.zeros(24),
            time=[datetime.datetime(2000, 1, 1, n) for n in range(24)],
            lon=np.linspace(-180, 180, 24),
            lat=np.linspace(-90, 90, 24),
        )
    )


@pytest.fixture()
def tracks_0_360():
    return xr.Dataset(
        dict(
            track_id=np.zeros(24),
            time=[datetime.datetime(2000, 1, 1, n) for n in range(24)],
            lon=np.linspace(-180, 180, 24) % 360,
            lat=np.linspace(-90, 90, 24),
        )
    )


coords = namedtuple("coords", ["lon", "lat"])


@pytest.fixture()
def tracks_as_list():
    return coords(lon=[0, 20], lat=[0, 0])


@pytest.fixture()
def tracks_as_point():
    return coords(lon=0.0, lat=0.0)
