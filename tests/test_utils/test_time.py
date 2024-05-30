import numpy as np

import huracanpy


def test_seasons():
    data = huracanpy.load(huracanpy.example_year_file)
    season = huracanpy.utils.time.get_season(data.track_id, data.lat, data.time)
    assert season.str.len().min() == 4
    assert season.str.len().max() == 8
    assert season.astype(int).min() == 1996
    assert season.astype(int).max() == 19961997
    np.testing.assert_approx_equal(season.astype(int).mean(), 7039001.37598945, 1)
