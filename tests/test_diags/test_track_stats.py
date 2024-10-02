import huracanpy

import numpy as np


def test_ace(tracks_csv):
    ace = huracanpy.diags.ace_by_track(tracks_csv, tracks_csv.wind10)

    np.testing.assert_allclose(ace, np.array([3.03623809, 2.21637375, 4.83686787]))

    assert isinstance(ace.data, np.ndarray)


def test_pace(tracks_csv):
    # Pass wind values to fit a (quadratic) model to the pressure-wind relationship
    pace, model = huracanpy.diags.pace_by_track(
        tracks_csv, tracks_csv.slp, wind=tracks_csv.wind10
    )

    np.testing.assert_allclose(pace, np.array([4.34978137, 2.65410482, 6.09892875]))

    # Call with the already fit model instead of wind values
    pace, _ = huracanpy.diags.pace_by_track(
        tracks_csv,
        tracks_csv.slp,
        model=model,
    )

    np.testing.assert_allclose(pace, np.array([4.34978137, 2.65410482, 6.09892875]))


def test_duration():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    d = huracanpy.diags.duration(data.time, data.track_id)
    assert d.min() == 126
    assert d.max() == 324
    assert d.mean() == 210


def test_gen_vals():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    G = huracanpy.diags.gen_vals(data)
    assert G.day.mean() == 10


def test_extremum_vals():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    M = huracanpy.diags.extremum_vals(data, "wind10", "max")
    m = huracanpy.diags.extremum_vals(data, "slp", "min")
    assert M.day.mean() == 15
    assert m.lat.mean() == -27
