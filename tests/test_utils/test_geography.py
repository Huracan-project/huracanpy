import pytest

import numpy as np

import huracanpy


@pytest.mark.parametrize(
    "data, expected",
    [
        ("tracks_minus180_plus180", np.array(["S"] * 12 + ["N"] * 12)),
        ("tracks_0_360", np.array(["S"] * 12 + ["N"] * 12)),
        ("tracks_csv", np.array(["S"] * 99)),
    ],
)
def test_hemisphere(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.utils.geography.get_hemisphere(data.lat)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            "tracks_minus180_plus180",
            np.array(["SP"] * 8 + ["SA"] * 4 + ["MED"] * 2 + ["NI"] * 4 + ["WNP"] * 6),
        ),
        (
            "tracks_0_360",
            np.array(["SP"] * 8 + ["SA"] * 4 + ["MED"] * 2 + ["NI"] * 4 + ["WNP"] * 6),
        ),
        ("tracks_csv", np.array(["AUS"] * 51 + ["SI"] * 48)),
    ],
)
def test_basin(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.utils.geography.get_basin(data.lon, data.lat)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            "tracks_minus180_plus180",
            np.array(
                ["Land"]
                + ["Ocean"] * 6
                + ["Land"] * 2
                + ["Ocean"] * 4
                + ["Land"]
                + ["Ocean"]
                + ["Land"] * 6
                + ["Ocean"] * 3
            ),
        ),
        (
            "tracks_0_360",
            np.array(
                ["Land"]
                + ["Ocean"] * 6
                + ["Land"] * 2
                + ["Ocean"] * 4
                + ["Land"]
                + ["Ocean"]
                + ["Land"] * 6
                + ["Ocean"] * 3
            ),
        ),
        ("tracks_csv", np.array(["Ocean"] * 15 + ["Land"] * 15 + ["Ocean"] * 69)),
    ],
)
def test_get_land_ocean(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.utils.geography.get_land_or_ocean(data.lon, data.lat)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            "tracks_minus180_plus180",
            np.array(
                ["Antarctica"]
                + [""] * 6
                + ["Argentina"] * 2
                + [""] * 4
                + ["Sudan"]
                + [""]
                + ["Iran"]
                + ["Afghanistan"]
                + ["China"]
                + ["Mongolia"]
                + ["Russia"] * 2
                + [""] * 3
            ),
        ),
        (
            "tracks_0_360",
            np.array(
                ["Antarctica"]
                + [""] * 6
                + ["Argentina"] * 2
                + [""] * 4
                + ["Sudan"]
                + [""]
                + ["Iran"]
                + ["Afghanistan"]
                + ["China"]
                + ["Mongolia"]
                + ["Russia"] * 2
                + [""] * 3
            ),
        ),
        ("tracks_csv", np.array([""] * 15 + ["Australia"] * 15 + [""] * 69)),
    ],
)
def test_get_country(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.utils.geography.get_country(data.lon, data.lat)
    assert (result == expected).all()
