import pytest

import huracanpy


@pytest.mark.parametrize("baselon", [-180, 0])
@pytest.mark.parametrize("method", ["histogram"])
@pytest.mark.parametrize("crop", [True, False])
def test_density(baselon, method, crop):
    data = huracanpy.load(huracanpy.example_year_file, baselon=baselon)
    D = huracanpy.calc.density(data.lon, data.lat, method=method, crop=crop)
    assert D.min() == 0.0
    assert D.max() == 43.0
    assert D.sum() == len(data.record)


def test_track_density_fails():
    data = huracanpy.load(huracanpy.example_year_file)
    with pytest.raises(NotImplementedError, match="Method nonsense not implemented"):
        huracanpy.calc.density(data.lon, data.lat, method="nonsense")
