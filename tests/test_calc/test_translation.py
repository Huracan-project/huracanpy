import numpy as np
from metpy.units import units

import huracanpy


def test_get_distance():
    data = huracanpy.load(huracanpy.example_csv_file)

    dist_geod = huracanpy.calc.distance(data.lon, data.lat, data.track_id)
    dist_haversine = huracanpy.calc.distance(
        data.lon, data.lat, data.track_id, method="haversine"
    )

    np.testing.assert_approx_equal(dist_geod[0], 170895, significant=6)
    np.testing.assert_approx_equal(dist_haversine[0], 170782, significant=6)
    assert (dist_haversine - dist_geod).max() < 1500 * units.m


def test_get_translation_speed():
    data = huracanpy.load(huracanpy.example_csv_file)

    ts = huracanpy.calc.translation_speed(data.lon, data.lat, data.time, data.track_id)

    np.testing.assert_approx_equal(ts[0], 7.9, significant=2)
    np.testing.assert_approx_equal(ts.mean(), 6.04, significant=3)
