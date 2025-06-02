import numpy as np
import pytest

import huracanpy


@pytest.mark.parametrize("lat_radians", [True, False])
@pytest.mark.parametrize("rmw_units", ["m", "km"])
@pytest.mark.parametrize("dequantify_rmw", [True, False])
def test_radius_of_maximum_wind(lat_radians, dequantify_rmw, rmw_units):
    data = huracanpy.load(huracanpy.example_TRACK_file, source="TRACK")
    rmw = huracanpy.calc.distance(
        data.lon, data.lat, data.feature_9_lon, data.feature_9_lat, method="haversine"
    )

    rmw = rmw.metpy.convert_units(rmw_units).metpy.dequantify()
    if dequantify_rmw:
        del rmw.attrs["units"]
    if lat_radians:
        lat = np.deg2rad(data.lat)
    else:
        lat = data.lat

    V_drift, theta_drift = huracanpy.tc.beta_drift(lat, data.feature_9, rmw)
    np.testing.assert_allclose(V_drift.mean(), 4.50560945)
    np.testing.assert_allclose(theta_drift.mean(), 330.379092)
