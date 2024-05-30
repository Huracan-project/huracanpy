import numpy as np

import huracanpy


def test_duration():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    d = huracanpy.diags.track_stats.duration(data)
    assert np.timedelta64(d.min().values, "h") == np.timedelta64(126, "h")
    assert np.timedelta64(d.max().values, "h") == np.timedelta64(324, "h")
    assert np.timedelta64(d.mean().values, "h") == np.timedelta64(210, "h")


def test_gen_vals():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    G = huracanpy.diags.track_stats.gen_vals(data)
    assert G.day.mean() == 10


def test_extremum_vals():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    M = huracanpy.diags.track_stats.extremum_vals(data, "wind10", "max")
    m = huracanpy.diags.track_stats.extremum_vals(data, "slp", "min")
    assert M.day.mean() == 15
    assert m.lat.mean() == -27
