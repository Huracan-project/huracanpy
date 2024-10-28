import huracanpy

import numpy as np


def test_climato():
    data = huracanpy.load(
        huracanpy.example_csv_file,
    )

    # get_freq
    freq = huracanpy.diags.get_freq(data.track_id)
    assert freq == 3

    # get_tc_days
    tc_days = huracanpy.diags.get_tc_days(time=data.time, track_ids=data.track_id)
    np.testing.assert_approx_equal(tc_days, 26.25, significant=4)

    # get_ace
    ace = huracanpy.diags.get_ace(data.wind10)
    np.testing.assert_approx_equal(ace, 10.0894797, significant=6)
