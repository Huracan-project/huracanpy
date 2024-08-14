import numpy as np

import huracanpy


def test_seasons():
    data = huracanpy.load(huracanpy.example_year_file)
    season = huracanpy.utils.time.get_season(data.track_id, data.lat, data.time)
    assert season.astype(int).min() == 1996
    assert season.astype(int).max() == 1997
    np.testing.assert_approx_equal(season.astype(int).mean(), 1996.09894459, 1e-6)
