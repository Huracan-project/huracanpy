import numpy as np
from metpy.units import units

import huracanpy


def test_get_delta():
    data = huracanpy.load(huracanpy.example_csv_file)

    # Test with only delta_var
    ## With wind
    delta_wind = huracanpy.utils.get_delta(data.wind10)
    np.testing.assert_approx_equal(delta_wind.mean(), 0.089352551, significant=6)
    ## With slp
    delta_slp = huracanpy.utils.get_delta(data.slp)
    np.testing.assert_approx_equal(delta_slp.mean(), -23.8743878, significant=6)

    # Test with track_ids
    delta_wind = huracanpy.utils.get_delta(data.wind10, data.track_id)
    assert np.isnan(delta_wind).sum() == len(np.unique(data.track_id))

    # Test centering options
    delta_wind = huracanpy.utils.get_delta(
        data.wind10, data.track_id, centering="forward"
    )
    assert len(delta_wind) == len(data.wind10)
    assert np.isnan(delta_wind[-1])
    delta_wind = huracanpy.utils.get_delta(
        data.wind10, data.track_id, centering="backward"
    )
    assert len(delta_wind) == len(data.wind10)
    assert np.isnan(delta_wind[0])

    # Test units
    delta_wind = huracanpy.utils.get_delta(data.wind10, var_units="m/s")
    delta_slp = huracanpy.utils.get_delta(data.slp, var_units="hPa")

    ## TODO: Test for time


def test_get_rate():
    data = huracanpy.load(huracanpy.example_csv_file)

    intensification_rate_wind = huracanpy.utils.get_rate(
        data.wind10,
        data.time,
        data.track_id,
        var_units="m/s",
    )
    np.testing.assert_approx_equal(
        intensification_rate_wind.mean(), 2.76335962e-06, significant=6
    )
    assert intensification_rate_wind.metpy.units == units("m/s^2")

    intensification_rate_slp = huracanpy.utils.get_rate(
        data.slp,
        data.time,
        data.track_id,
        var_units="hPa",
    ).metpy.convert_units("hectopascals/hour")
    np.testing.assert_approx_equal(
        intensification_rate_slp.min(), -124.115, significant=6
    )
