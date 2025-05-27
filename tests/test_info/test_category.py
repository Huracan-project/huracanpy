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
