import huracanpy


def test_simple_track_density():
    data = huracanpy.load(huracanpy.example_year_file)
    D = huracanpy.diags.simple_global_histogram(data.lon, data.lat)
    assert D.min() == 0.0
    assert D.max() == 43.0
    assert D.sum() == len(data.record)
