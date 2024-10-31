import pytest
import numpy as np

import huracanpy


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
