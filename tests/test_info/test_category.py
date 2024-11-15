import numpy as np

import huracanpy


def test_get_category():
    data = huracanpy.load(huracanpy.example_csv_file)

    # Test with made up data for each category
    result = huracanpy.info.category(
        data.wind10, bins=[0, 10, 20, 30], labels=[0, 1, 2]
    )

    # Separate test for last value (nan)
    assert result.min() == 1
    assert result.max() == 2
    assert result.sum() == 59 + 2 * 40


def test_category_strings(tracks_csv):
    result = huracanpy.info.category(
        tracks_csv.wind10, bins=[0, 10, 20, 30], labels=["low", "med", "high"]
    )

    assert np.count_nonzero(result == "low") == 0
    assert np.count_nonzero(result == "med") == 59
    assert np.count_nonzero(result == "high") == 40
