import pytest
import numpy as np

import huracanpy


def test_time_components(tracks_csv):
    year, month, day, hour = huracanpy.info.time_components(tracks_csv.time)

    np.testing.assert_equal(year, np.array([1980] * len(tracks_csv.time)))
    np.testing.assert_equal(month, np.array([1] * len(tracks_csv.time)))

    days = np.concatenate(
        [
            # First track
            np.array([6] * 3),
            np.concatenate([np.array([n] * 4) for n in range(7, 13 + 1)]),
            # Second track
            np.concatenate([np.array([n] * 4) for n in range(7, 8 + 1)]),
            np.array([9] * 2),
            np.concatenate([np.array([n] * 4) for n in range(10, 11 + 1)]),
            np.array([12] * 2),
            # Third track
            np.array([17, 17, 18, 19, 19, 20, 20, 20]),
            np.concatenate([np.array([n] * 4) for n in range(21, 30 + 1)]),
        ]
    )
    np.testing.assert_equal(day, days)

    hours = np.concatenate(
        [
            # First track
            np.array([6, 12, 18, 0] * 8)[:-1],
            # Second track
            np.array([0, 6, 12, 18] * 2),
            np.array([6, 18]),
            np.array([0, 6, 12, 18] * 3)[:-2],
            # Third track
            np.array([6, 18, 6, 0, 6, 6, 12, 18]),
            np.array([0, 6, 12, 18] * 10),
        ]
    )
    np.testing.assert_equal(hour, hours)


@pytest.mark.parametrize(("tracks",), [("tracks_year",), ("tracks_year_cftime",)])
def test_seasons(tracks, request):
    tracks = request.getfixturevalue(tracks)

    season = huracanpy.info.season(tracks.track_id, tracks.lat, tracks.time)
    assert season.astype(int).min() == 1996
    assert season.astype(int).max() == 1997
    np.testing.assert_approx_equal(season.astype(int).mean(), 1996.09894459, 1e-6)


def test_season_long(tracks_year):
    season = huracanpy.info.season(
        tracks_year.track_id, tracks_year.lat, tracks_year.time, convention="tc-long"
    )

    season_nh = season[tracks_year.lat >= 0].astype(int)
    assert (season_nh.astype(int) == 1996).all()

    season_sh = season[tracks_year.lat < 0].astype(int)
    assert season_sh.min() == 19951996
    assert season_sh.max() == 19961997
    np.testing.assert_approx_equal(season_sh.mean(), 19954801.76)


def test_season_fails(tracks_year):
    with pytest.raises(NotImplementedError, match="Convention not recognized"):
        huracanpy.info.season(
            tracks_year.track_id,
            tracks_year.lat,
            tracks_year.time,
            convention="nonsense",
        )
