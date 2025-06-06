import pytest

import huracanpy


def test_duration():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    d = huracanpy.calc.track_duration(data.time, data.track_id)
    assert d.min() == 126
    assert d.max() == 324
    assert d.mean() == 210


def test_gen_vals():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    g = huracanpy.calc.gen_vals(data, data.time, data.track_id)
    assert g.time.dt.day.mean() == 10


def test_apex_vals():
    data = huracanpy.load(huracanpy.example_csv_file)
    max_wind = huracanpy.calc.apex_vals(data, data.wind10, data.track_id, "max")
    min_pressure = huracanpy.calc.apex_vals(data, data.slp, data.track_id, "min")
    assert max_wind.time.dt.day.mean() == 15
    assert min_pressure.lat.mean() == -27


def test_apex_vals_fails(tracks_csv):
    with pytest.raises(NotImplementedError, match="stat not recognized"):
        huracanpy.calc.apex_vals(
            tracks_csv, tracks_csv.wind10, tracks_csv.track_id, "nonsense"
        )
