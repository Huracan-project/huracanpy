from metpy.units import units
import numpy as np
import pytest

import huracanpy


@pytest.mark.parametrize("with_units", [True, False])
@pytest.mark.parametrize("with_units_on_bins", [True, False])
def test_get_category(tracks_csv, with_units, with_units_on_bins, bins=[0, 10, 20, 30]):
    if with_units:
        tracks_csv.wind10.attrs["units"] = "m s-1"
    if with_units_on_bins:
        bins = units.Quantity(bins, "m s-1")

    # Test with made up data for each category
    result = huracanpy.info.category(tracks_csv.wind10, bins=bins, labels=[0, 1, 2])

    # Separate test for last value (nan)
    assert result.min() == 1
    assert result.max() == 2
    assert result.sum() == 59 + 2 * 40
    assert isinstance(result.data, np.ndarray)


def test_category_warns(tracks_csv):
    with pytest.warns(UserWarning, match="labels not provided"):
        huracanpy.info.category(tracks_csv.wind10, bins=[0, 10, 20, 30])


def test_category_strings(tracks_csv):
    result = huracanpy.info.category(
        tracks_csv.wind10, bins=[0, 10, 20, 30], labels=["low", "med", "high"]
    )

    assert np.count_nonzero(result == "low") == 0
    assert np.count_nonzero(result == "med") == 59
    assert np.count_nonzero(result == "high") == 40


_expected = np.array(
    [
        7,
        6,
        6,
        8,
        7,
        7,
        8,
        9,
        10,
        9,
        9,
        10,
        10,
        10,
        10,
        10,
        10,
        9,
        9,
        7,
        7,
        7,
        8,
        7,
        7,
        6,
        6,
        6,
        6,
        6,
        5,
        6,
        8,
        8,
        8,
        8,
        8,
        9,
        9,
        9,
        9,
        9,
        8,
        8,
        8,
        8,
        7,
        7,
        6,
        6,
        5,
        6,
        6,
        7,
        7,
        6,
        5,
        6,
        6,
        6,
        7,
        6,
        7,
        6,
        6,
        6,
        7,
        6,
        6,
        6,
        6,
        8,
        7,
        8,
        8,
        8,
        8,
        8,
        9,
        9,
        9,
        8,
        9,
        9,
        9,
        9,
        9,
        9,
        10,
        10,
        10,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
    ]
)


def test_beaufort(tracks_csv):
    result = huracanpy.info.beaufort_category(tracks_csv.wind10)

    np.testing.assert_array_equal(result, _expected)
