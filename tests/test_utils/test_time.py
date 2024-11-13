import numpy as np
import pytest

import huracanpy


@pytest.mark.parametrize(("tracks",), [("tracks_year",), ("tracks_year_cftime",)])
def test_seasons(tracks, request):
    tracks = request.getfixturevalue(tracks)

    season = huracanpy.utils.time.get_season(tracks.track_id, tracks.lat, tracks.time)
    assert season.astype(int).min() == 1996
    assert season.astype(int).max() == 1997
    np.testing.assert_approx_equal(season.astype(int).mean(), 1996.09894459, 1e-6)
