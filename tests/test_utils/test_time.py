import numpy as np

import huracanpy


def test_get_time():
    data = huracanpy.load(huracanpy.example_csv_file)

    # get_time
    time = huracanpy.utils.time.get_time(data.year, data.month, data.day, data.hour)
    assert time.isel(record=0) == np.datetime64("1980-01-06 06:00:00")

    # get_time_components
    year, month, day, hour = huracanpy.utils.time.get_time_components(data.time)
    assert (
        year.isel(record=0),
        month.isel(record=0),
        day.isel(record=0),
        hour.isel(record=0),
    ) == (1980, 1, 6, 6)


def test_seasons():
    data = huracanpy.load(huracanpy.example_year_file)
    season = huracanpy.utils.get_season(data.track_id, data.lat, data.time)
    assert season.astype(int).min() == 1996
    assert season.astype(int).max() == 1997
    np.testing.assert_approx_equal(season.astype(int).mean(), 1996.09894459, 1e-6)
