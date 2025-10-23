import pytest

import huracanpy
import numpy as np


@pytest.mark.parametrize("baselon", [-180, 0])
@pytest.mark.parametrize("method", ["histogram"])
@pytest.mark.parametrize("crop", [True, False])
def test_density(baselon, method, crop):
    data = huracanpy.load(huracanpy.example_year_file, baselon=baselon)
    d = huracanpy.calc.density(data.lon, data.lat, method=method, crop=crop)
    assert d.min() == 0.0
    assert d.max() == 43.0
    assert d.sum() == len(data.record)

    # "crop=True" used to cut out any rows/columns with all NaN (no data) but this leads
    # to non-even spacing in longitude or latitude where it has cut out in between data
    # Instead, we only want to crop around the outside of the valid data
    assert (np.diff(d.lon) == 5).all()
    assert (np.diff(d.lat) == 5).all()


def test_track_density_fails():
    data = huracanpy.load(huracanpy.example_year_file)
    with pytest.raises(NotImplementedError, match="Method nonsense not implemented"):
        huracanpy.calc.density(data.lon, data.lat, method="nonsense")
