import numpy as np

import huracanpy

def test_radius_of_maximum_wind():
    data = huracanpy.load(huracanpy.example_TRACK_file, source = "TRACK")
    RMW = huracanpy.tc.radius_of_maximum_wind(data.lon, data.lat, data.feature_9_lon, data.feature_9_lat)
    V_drift, theta_drift = huracanpy.tc.beta_drift(data.lat, data.feature_9, RMW)
    np.testing.assert_almost_equal(V_drift.mean(), 4.50333266)
    np.testing.assert_almost_equal(theta_drift.mean(), 330.38964024)