import huracanpy


def test_duration():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    d = huracanpy.diags.get_track_duration(data.time, data.track_id)
    assert d.min() == 126
    assert d.max() == 324
    assert d.mean() == 210


def test_gen_vals():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    G = huracanpy.diags.get_gen_vals(data)
    assert G.time.dt.day.mean() == 10


def test_extremum_vals():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    M = huracanpy.diags.get_apex_vals(data, "wind10", "max")
    m = huracanpy.diags.get_apex_vals(data, "slp", "min")
    assert M.time.dt.day.mean() == 15
    assert m.lat.mean() == -27
