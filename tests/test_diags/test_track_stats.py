import huracanpy


def test_duration():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    d = huracanpy.diags.track_stats.duration(data)
    assert d.min() == 126
    assert d.max() == 324
    assert d.mean() == 210


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
