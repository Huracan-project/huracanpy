import pytest

import huracanpy

import numpy as np


@pytest.mark.parametrize(
    "aggregate_by, result",
    [(None, [10.0894797]), ("track_id", [3.03623809, 2.21637375, 4.83686787])],
)
def test_ace(aggregate_by, result):
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")

    if aggregate_by is not None:
        aggregate_by = data[aggregate_by]
    ace = huracanpy.tc.ace(data.wind10, aggregate_by=aggregate_by)
    if aggregate_by is None:
        ace = ace.sum()

    np.testing.assert_allclose(ace, result)

    assert isinstance(ace.data, np.ndarray)


def test_pace():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    # Pass wind values to fit a (quadratic) model to the pressure-wind relationship
    pace, model = huracanpy.tc.pace(
        data.slp,
        data.wind10,
        aggregate_by=data.track_id,
    )

    np.testing.assert_allclose(pace, np.array([4.34978137, 2.65410482, 6.09892875]))

    # Call with the already fit model instead of wind values
    pace, _ = huracanpy.tc.pace(
        data.slp,
        model=model,
        aggregate_by=data.track_id,
    )

    np.testing.assert_allclose(pace, np.array([4.34978137, 2.65410482, 6.09892875]))


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
