import numpy as np

import huracanpy


def test_simple_track_density():
    data = huracanpy.load(huracanpy.example_year_file)
    D = huracanpy.diags.track_density.simple_global_histogram(data.lon, data.lat)
    assert D.min() == 1.0
    assert D.max() == 43.0
    assert D.median() == 4.0
    assert np.isnan(D).sum() == 2240
