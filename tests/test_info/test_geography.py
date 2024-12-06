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
    result = huracanpy.info.hemisphere(data.lat)
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
    result = huracanpy.info.basin(data.lon, data.lat)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "convention, expected",
    [
        ("Sainsbury2022JCLI", ["WEST"] + ["SUB", "MDR"] * 6 + ["SUB"] + [""] * 6),
        (
            "Sainsbury2022MWR",
            [""] * 3 + ["NoEurope", ""] * 6 + ["Europe", ""] * 2 + ["Europe"],
        ),
    ],
)
def test_basin_definition(convention, expected):
    lon = np.arange(-80, 20, 5)
    lat = np.array([15, 45] * 10)
    result = huracanpy.info.basin(lon, lat, convention=convention)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            "tracks_minus180_plus180",
            np.array(
                [False]
                + [True] * 6
                + [False] * 2
                + [True] * 4
                + [False]
                + [True]
                + [False] * 6
                + [True] * 3
            ),
        ),
        (
            "tracks_0_360",
            np.array(
                [False]
                + [True] * 6
                + [False] * 2
                + [True] * 4
                + [False]
                + [True]
                + [False] * 6
                + [True] * 3
            ),
        ),
        ("tracks_csv", np.array([True] * 15 + [False] * 15 + [True] * 69)),
    ],
)
def test_get_land_ocean(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.info.is_ocean(data.lon, data.lat)
    result_land = huracanpy.info.is_land(data.lon, data.lat)

    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(~result_land, expected)


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
    result = huracanpy.info.country(data.lon, data.lat)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            "tracks_minus180_plus180",
            np.array(
                ["Antarctica"]
                + [""] * 6
                + ["South America"] * 2
                + [""] * 4
                + ["Africa"]
                + [""]
                + ["Asia"] * 4
                + ["Europe"] * 2
                + [""] * 3
            ),
        ),
        (
            "tracks_0_360",
            np.array(
                ["Antarctica"]
                + [""] * 6
                + ["South America"] * 2
                + [""] * 4
                + ["Africa"]
                + [""]
                + ["Asia"] * 4
                + ["Europe"] * 2
                + [""] * 3
            ),
        ),
        ("tracks_csv", np.array([""] * 15 + ["Oceania"] * 15 + [""] * 69)),
    ],
)
def test_get_continent(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.info.continent(data.lon, data.lat)
    assert (result == expected).all()
