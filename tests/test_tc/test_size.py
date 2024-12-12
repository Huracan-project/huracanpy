import numpy as np

import huracanpy

def test_radius_of_maximum_wind():
    data = huracanpy.load(huracanpy.example_TRACK_file, source = "TRACK")
    RMW = huracanpy.tc.radius_of_maximum_wind(data.lon, data.lat, data.feature_9_lon, data.feature_9_lat)
    np.testing.assert_almost_equal(RMW.max(), 666.74927689)
    np.testing.assert_almost_equal(RMW.min(), 25.21889771)
    np.testing.assert_almost_equal(RMW.mean(), 356.36039091)