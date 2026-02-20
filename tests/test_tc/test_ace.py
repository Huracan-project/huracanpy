from metpy.units import units
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

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
    "model, threshold_pressure, threshold_wind, sum_by, kwargs, result",
    [
        (None, None, None, "track_id", {}, [4.34978137, 2.65410482, 6.09892875]),
        # Passing the Polynomial explicitly gives the same answer
        (
            Polynomial,
            None,
            None,
            "track_id",
            {"deg": 2},
            [4.34978137, 2.65410482, 6.09892875],
        ),
        # Recreating an equivalent polynomial fit with sklearn gives the same answer
        (
            make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
            ),
            None,
            None,
            "track_id",
            {},
            [4.34978137, 2.65410482, 6.09892875],
        ),
        (None, 1e5, None, "track_id", {}, [4.115545, 2.654105, 5.152805]),
        (
            None,
            None,
            17 * units("m s-1"),
            "track_id",
            {},
            [3.332754, 2.281562, 4.49114],
        ),
        ("z2021", None, None, "track_id", {}, [2.266398, 1.42874, 3.19211]),
        ("holland", None, None, "track_id", {}, [2.03433554, 1.23160966, 2.72186645]),
    ],
)
def test_pace(
    tracks_csv, model, threshold_pressure, threshold_wind, sum_by, kwargs, result
):
    if sum_by is not None:
        sum_by = tracks_csv[sum_by]

    # Pass wind values to fit a (quadratic) model to the pressure-wind relationship
    pace, model = huracanpy.tc.pace(
        tracks_csv.slp,
        tracks_csv.wind10,
        model=model,
        sum_by=sum_by,
        threshold_wind=threshold_wind,
        threshold_pressure=threshold_pressure,
        pressure_units="Pa",
        **kwargs,
    )

    np.testing.assert_allclose(pace, result)

    # Call with the already fit model instead of wind values
    pace, _ = huracanpy.tc.pace(
        tracks_csv.slp,
        model=model,
        sum_by=sum_by,
        threshold_wind=threshold_wind,
        threshold_pressure=threshold_pressure,
        pressure_units="Pa",
        **kwargs,
    )

    np.testing.assert_allclose(pace, result)


def test_pace_fails(tracks_csv):
    with pytest.raises(ValueError, match="Need to specify either wind or model"):
        huracanpy.tc.pace(tracks_csv.slp)
