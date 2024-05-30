import pytest
import numpy as np
import xarray as xr

import huracanpy


@pytest.mark.parametrize(
    "convention, data, expected",
    [
        (
            "Saffir-Simpson",
            np.array([-1e24, 0, 20, 30, 40, 50, 60, 70, 1e24, np.nan]),
            np.array([-1, -1, 0, 1, 2, 3, 4, 5, 5, np.nan]),
        ),
        (
            "Klotzbach",
            np.array(
                [1e24, 1006, 1000, 985, 971, 961, 950, 930, 921, 900, 1e-24, -1, np.nan]
            ),
            np.array([-1, -1, 0, 1, 2, 2, 3, 4, 5, 5, 5, 5, np.nan]),
        ),
        (
            "Simpson",
            np.array(
                [1e24, 1006, 1000, 985, 971, 961, 950, 930, 921, 900, 1e-24, -1, np.nan]
            ),
            np.array([-1, -1, -1, 0, 1, 3, 3, 4, 4, 5, 5, 5, np.nan]),
        ),
    ],
)
def test_categorise(convention, data, expected):
    # Test with made up data for each category
    result = huracanpy.utils.category.categorize(data, convention=convention)

    # Separate test for last value (nan)
    assert (result[:-1] == expected[:-1]).all()
    assert np.isnan(result[-1])


def test_sshs():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    assert huracanpy.utils.category.get_sshs_cat(data.wind10).min() == -1
    assert huracanpy.utils.category.get_sshs_cat(data.wind10).max() == 0


def test_pressure_cat():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    Klotz = huracanpy.utils.category.get_pressure_cat(data.slp / 100)
    Simps = huracanpy.utils.category.get_pressure_cat(
        data.slp / 100, convention="Simpson"
    )
    assert Klotz.sum() == 62
    assert Simps.sum() == -23


@pytest.mark.parametrize("pass_as_numpy", [False, True])
@pytest.mark.parametrize(
    "units, expected",
    [
        ("m s-1", "default"),
        ("cm s-1", np.array([-1.0] * 99)),
        ("km s-1", np.array([5.0] * 99)),
    ],
)
def test_sshs_units(units, expected, pass_as_numpy):
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")

    if isinstance(expected, str) and expected == "default":
        expected = huracanpy.utils.category.get_sshs_cat(data.wind10)

    if pass_as_numpy:
        result = huracanpy.utils.category.get_sshs_cat(
            data.wind10.data, wind_units=units
        )
    else:
        data.wind10.attrs["units"] = units
        result = huracanpy.utils.category.get_sshs_cat(data.wind10)

    (result == expected).all()

    if pass_as_numpy:
        assert isinstance(result, np.ndarray)
    else:
        assert isinstance(result, xr.DataArray)


@pytest.mark.parametrize("pass_as_numpy", [False, True])
@pytest.mark.parametrize("convention", ["Klotzbach", "Simpson"])
@pytest.mark.parametrize(
    "units, expected",
    [
        ("Pa", "default"),
        ("hPa", np.array([-1.0] * 99)),
    ],
)
def test_pressure_cat_units(units, expected, convention, pass_as_numpy):
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")

    if isinstance(expected, str) and expected == "default":
        expected = huracanpy.utils.category.get_pressure_cat(
            data.slp, convention=convention
        )

    if pass_as_numpy:
        result = huracanpy.utils.category.get_pressure_cat(
            data.slp.data, convention=convention, slp_units=units
        )
    else:
        data.slp.attrs["units"] = units
        result = huracanpy.utils.category.get_pressure_cat(
            data.slp, convention=convention
        )

    (result == expected).all()

    if pass_as_numpy:
        assert isinstance(result, np.ndarray)
    else:
        assert isinstance(result, xr.DataArray)
