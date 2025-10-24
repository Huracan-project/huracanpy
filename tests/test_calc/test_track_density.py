import pytest

import huracanpy
from metpy.constants import earth_avg_radius
import numpy as np


@pytest.mark.parametrize("baselon", [-180, 0])
@pytest.mark.parametrize("method", ["histogram", "kde"])
@pytest.mark.parametrize("crop", [True, False])
def test_density(baselon, method, crop):
    data = huracanpy.load(huracanpy.example_year_file, baselon=baselon)
    d = huracanpy.calc.density(data.lon, data.lat, method=method, crop=crop)

    # Currently non-spherical KDE gives inconsistent results
    if method != "kde":
        assert d.min() == 0.0
        assert d.max() == 43.0
    np.testing.assert_allclose(d.sum(), len(data.record))

    # "crop=True" used to cut out any rows/columns with all NaN (no data) but this leads
    # to non-even spacing in longitude or latitude where it has cut out in between data
    # Instead, we only want to crop around the outside of the valid data
    assert (np.diff(d.lon) == 5).all()
    assert (np.diff(d.lat) == 5).all()


def test_density_spherical():
    data = huracanpy.load(huracanpy.example_year_file)

    d = huracanpy.calc.density(data.lon, data.lat)
    d_spherical = huracanpy.calc.density(data.lon, data.lat, spherical=True)

    dlon = d.lon[1] - d.lon[0]
    dlat = d.lat[1] - d.lat[0]
    x_edge = np.arange(d.lon[0] - dlon * 0.5, d.lon[-1] + dlon, dlon)
    y_edge = np.arange(d.lat[0] - dlat * 0.5, d.lat[-1] + dlat, dlat)
    area = (earth_avg_radius.magnitude**2) * np.outer(
        np.diff(np.sin(np.deg2rad(y_edge))), np.diff(np.deg2rad(x_edge))
    )

    # Check that surface area of Earth is approx correct
    np.testing.assert_allclose(area.sum(), 5.1e14, rtol=0.01)

    np.testing.assert_allclose(d, d_spherical * area)

    assert d_spherical.attrs["units"] == "1 / meter ** 2"
    assert "units" not in d.attrs


@pytest.mark.parametrize("baselon", [-180, 0])
@pytest.mark.parametrize("crop", [True, False])
def test_density_spherical_kde(baselon, crop):
    data = huracanpy.load(huracanpy.example_year_file, baselon=baselon)

    d = huracanpy.calc.density(
        data.lon, data.lat, method="kde", spherical=True, crop=crop
    )

    assert d.attrs["units"] == "1 / meter ** 2"

    np.testing.assert_allclose(d.min(), 3.65054709e-14)
    np.testing.assert_allclose(d.max(), 2.85069764e-11)
    np.testing.assert_allclose(d.sum(), 8.6133285e-09)


def test_track_density_fails():
    data = huracanpy.load(huracanpy.example_year_file)
    with pytest.raises(NotImplementedError, match="Method nonsense not implemented"):
        huracanpy.calc.density(data.lon, data.lat, method="nonsense")
