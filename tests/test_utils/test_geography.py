import numpy as np

import huracanpy


def test_hemisphere():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    assert np.unique(huracanpy.utils.geography.get_hemisphere(data.lat)) == np.array(
        ["S"]
    )


def test_basin():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    assert huracanpy.utils.geography.get_basin(data.lon, data.lat)[0] == "AUS"
    assert huracanpy.utils.geography.get_basin(data.lon, data.lat)[-1] == "SI"
