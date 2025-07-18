import numpy as np
import pint
import pytest
from metpy.units import units

import huracanpy


def test_azimuth():
    data = huracanpy.load(huracanpy.example_csv_file)
    az = huracanpy.calc.azimuth(data.lon, data.lat, data.track_id)

    np.testing.assert_approx_equal(az[0], -109.07454278, significant=6)
    np.testing.assert_approx_equal(az.mean(), 28.99955985, significant=6)


def test_azimuth_warns(tracks_csv):
    with pytest.warns(UserWarning, match="track_id is not provided"):
        huracanpy.calc.azimuth(tracks_csv.lon, tracks_csv.lat)


def test_get_distance():
    data = huracanpy.load(huracanpy.example_csv_file)

    dist_geod = huracanpy.calc.distance(data.lon, data.lat, data.track_id)
    dist_haversine = huracanpy.calc.distance(
        data.lon, data.lat, data.track_id, method="haversine"
    )

    np.testing.assert_approx_equal(dist_geod[0], 170895, significant=6)
    np.testing.assert_approx_equal(dist_haversine[0], 170782, significant=6)
    assert (dist_haversine - dist_geod).max() < 1500

    for dist in dist_haversine, dist_geod:
        assert not isinstance(dist.data, pint.Quantity)
        assert dist.metpy.units == units("m")


def test_distance_warns(tracks_csv):
    with pytest.warns(UserWarning, match="track_id is not provided"):
        huracanpy.calc.distance(tracks_csv.lon, tracks_csv.lat)


@pytest.mark.parametrize(
    "varnames, method, message",
    [
        (["lon", "lat", "lon"], "geod", "Distance either takes 2 arrays"),
        (["lon", "lat", "lon", "lat", "lon"], "geod", "Distance either takes 2 arrays"),
        (["lon", "lat"], "nonsense", "Method nonsense for distance calculation"),
    ],
)
def test_distance_fails(tracks_csv, varnames, method, message):
    with pytest.raises(ValueError, match=message):
        huracanpy.calc.distance(
            *[tracks_csv[varname] for varname in varnames],
            track_id=tracks_csv.track_id,
            method=method,
        )


def test_radius_of_maximum_wind():
    data = huracanpy.load(huracanpy.example_TRACK_file, source="TRACK")
    rmw = huracanpy.calc.distance(
        data.lon, data.lat, data.feature_9_lon, data.feature_9_lat, method="haversine"
    )
    np.testing.assert_allclose(rmw.max(), 666749.2768932)
    np.testing.assert_allclose(rmw.min(), 25218.89771)
    np.testing.assert_allclose(rmw.mean(), 356360.39091)


def test_get_translation_speed():
    data = huracanpy.load(huracanpy.example_csv_file)

    ts = huracanpy.calc.translation_speed(data.lon, data.lat, data.time, data.track_id)

    np.testing.assert_approx_equal(ts[0], 7.9, significant=2)
    np.testing.assert_approx_equal(ts.mean(), 6.04, significant=3)

    assert not isinstance(ts.data, pint.Quantity)
    assert ts.metpy.units == units("m s-1")


def test_translation_speed_warns(tracks_csv):
    with pytest.warns(UserWarning, match="track_id is not provided"):
        huracanpy.calc.translation_speed(
            tracks_csv.lon, tracks_csv.lat, tracks_csv.time
        )
