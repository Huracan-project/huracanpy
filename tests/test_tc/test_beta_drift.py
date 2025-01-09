import numpy as np

import huracanpy


def test_radius_of_maximum_wind():
    data = huracanpy.load(huracanpy.example_TRACK_file, source="TRACK")
    RMW = huracanpy.calc.distance(
        data.lon, data.lat, data.feature_9_lon, data.feature_9_lat, method="haversine"
    )
    V_drift, theta_drift = huracanpy.tc.beta_drift(data.lat, data.feature_9, RMW)
    np.testing.assert_allclose(V_drift.mean(), 4.50560945)
    np.testing.assert_allclose(theta_drift.mean(), 330.379092)
