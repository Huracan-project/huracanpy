from metpy.units import units
import numpy as np
import pytest

import huracanpy


@pytest.mark.parametrize(
    "threshold, sum_by, result",
    [
        (34 * units("knots"), None, [10.0894797]),
        ((34 * units("knots")).to("m s-1"), None, [10.0894797]),
        ((34 * units("knots")).to("m s-1").magnitude, None, [10.0894797]),
        (0, None, [13.23391871]),
        (34 * units("knots"), "track_id", [3.03623809, 2.21637375, 4.83686787]),
    ],
)
def test_ace(tracks_csv, threshold, sum_by, result):
    if sum_by is not None:
        sum_by = tracks_csv[sum_by]

    ace = huracanpy.tc.ace(tracks_csv.wind10, sum_by=sum_by, threshold=threshold)

    if sum_by is None:
        ace = ace.sum()

    np.testing.assert_allclose(ace, result)
    assert isinstance(ace.data, np.ndarray)


@pytest.mark.parametrize(
    "threshold_pressure, threshold_wind, sum_by, result",
    [
        (None, None, "track_id", [4.34978137, 2.65410482, 6.09892875]),
    ],
)
def test_pace(tracks_csv, threshold_pressure, threshold_wind, sum_by, result):
    if sum_by is not None:
        sum_by = tracks_csv[sum_by]

    # Pass wind values to fit a (quadratic) model to the pressure-wind relationship
    pace, model = huracanpy.tc.pace(
        tracks_csv.slp,
        tracks_csv.wind10,
        sum_by=sum_by,
        threshold_wind=threshold_wind,
        threshold_pressure=threshold_pressure,
    )

    np.testing.assert_allclose(pace, result)

    # Call with the already fit model instead of wind values
    pace, _ = huracanpy.tc.pace(
        tracks_csv.slp,
        model=model,
        sum_by=sum_by,
        threshold_wind=threshold_wind,
        threshold_pressure=threshold_pressure,
    )

    np.testing.assert_allclose(pace, np.array([4.34978137, 2.65410482, 6.09892875]))
