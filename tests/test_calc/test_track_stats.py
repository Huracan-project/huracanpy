import huracanpy


def test_duration():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    d = huracanpy.calc.get_track_duration(data.time, data.track_id)
    assert d.min() == 126
    assert d.max() == 324
    assert d.mean() == 210


def test_gen_vals():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    G = huracanpy.calc.get_gen_vals(data, data.time, data.track_id)
    assert G.time.dt.day.mean() == 10


def test_apex_vals():
    data = huracanpy.load(huracanpy.example_csv_file)
    M = huracanpy.calc.get_apex_vals(data, data.wind10, data.track_id, "max")
    m = huracanpy.calc.get_apex_vals(data, data.slp, data.track_id, "min")
    assert M.time.dt.day.mean() == 15
    assert m.lat.mean() == -27
