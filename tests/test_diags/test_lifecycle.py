import huracanpy


def test_time_from_genesis():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    t = huracanpy.diags.time_from_genesis(data)
    assert t.min() == 0
    assert t.max() == 1166400000000000


def test_time_from_extremum():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    t = huracanpy.diags.time_from_extremum(data, "wind10", "max")
    assert t.min() == -972000000000000
    assert t.max() == 367200000000000
