import pathlib

import numpy as np
import pint
import pytest
from metpy.units import units
import xarray as xr

import huracanpy


data_path = pathlib.Path(__file__).parent.parent / "saved_results"


def test_azimuth():
    data = huracanpy.load(huracanpy.example_csv_file)
    az = huracanpy.calc.azimuth(data.lon, data.lat, data.track_id)

    np.testing.assert_approx_equal(az[0], -109.07454278, significant=6)
    np.testing.assert_approx_equal(az.mean(), 28.99955985, significant=6)


@pytest.mark.parametrize("centering", ["centre", "adaptive"])
def test_azimuth_centred(centering):
    # Centred averaging of angles needs to account for circular nature
    # Use tracks that switch across -180/+180 to test the averaging
    lats = np.array(list(range(85, -91, -5)) + list(range(-85, 91, 5)))
    lons = np.array([-5, 5] * 17 + [-5, 0, 175] + [-175, 175] * 17 + [0])
    track_ids = np.zeros_like(lons)

    angles = huracanpy.calc.azimuth(
        lons, lats, track_ids, centering=centering
    ).magnitude

    # First angle south is the forward azimuth
    if centering == "centre":
        assert np.isnan(angles[0])
    else:
        np.testing.assert_approx_equal(angles[0], 160.365645)

    # Average direct south
    # TODO One angle is -180, which is equivalent to 180 but annoying that it is
    #  different
    # (5, -80) -> (0, -90)
    assert (np.abs(angles[1:36]) == 180).all()

    # First point after South pole points (from pole to point)
    assert angles[36] == -175

    # Average direct north
    assert (angles[37:-1] == 0).all()

    if centering == "centre":
        assert np.isnan(angles[-1])
    else:
        assert angles[-1] == 0


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


def test_corral_radius(tracks_csv):
    result = huracanpy.calc.corral_radius(
        tracks_csv.lon, tracks_csv.lat, tracks_csv.time, tracks_csv.track_id, window=36
    )

    expected = xr.open_dataarray(str(data_path / "corral_radius_result.nc"))
    np.testing.assert_allclose(result, expected.values, rtol=1e-4)


@pytest.mark.parametrize(
    "lons, lats, expected",
    [
        # Spherical cap. Tests treating distances across pole and dateline correctly
        ([0, 90, 180, 270], [80] * 4, [1107551.86696002] * 4),
        # Square around equator. Check dateline crossings are correct
        ([175, -175, 175, -175], [-5, -5, 5, 5], [784654.62840337] * 4),
        ([175, 185, 175, 185], [-5, -5, 5, 5], [784654.62840337] * 4),
        ([355, 5, 355, 5], [-5, -5, 5, 5], [784654.62840337] * 4),
        ([355, 365, 355, 365], [-5, -5, 5, 5], [784654.62840337] * 4),
    ],
)
def test_corral_radius_spherical(lons, lats, expected):
    result = huracanpy.calc.corral_radius(lons, lats)

    np.testing.assert_allclose(result, expected)
