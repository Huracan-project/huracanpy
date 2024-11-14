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
