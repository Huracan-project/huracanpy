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


def test_seasons():
    data = huracanpy.load(huracanpy.example_year_file)
    season = huracanpy.info.season(data.track_id, data.lat, data.time)
    assert season.astype(int).min() == 1996
    assert season.astype(int).max() == 1997
    np.testing.assert_approx_equal(season.astype(int).mean(), 1996.09894459, 1e-6)
