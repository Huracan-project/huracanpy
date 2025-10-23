import pathlib

import numpy as np
import pytest
import xarray as xr

import huracanpy


data_path = pathlib.Path(__file__).parent.parent / "saved_results"


@pytest.mark.parametrize(
    "data, expected",
    [
        ("tracks_minus180_plus180", np.asarray(["S"] * 12 + ["N"] * 12)),
        ("tracks_0_360", np.asarray(["S"] * 12 + ["N"] * 12)),
        ("tracks_csv", np.asarray(["S"] * 99)),
    ],
)
def test_hemisphere(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.info.hemisphere(data.lat)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            "tracks_minus180_plus180",
            np.asarray(
                ["SP"] * 8 + ["SA"] * 4 + ["MED"] * 2 + ["NI"] * 4 + ["WNP"] * 6
            ),
        ),
        (
            "tracks_0_360",
            np.asarray(
                ["SP"] * 8 + ["SA"] * 4 + ["MED"] * 2 + ["NI"] * 4 + ["WNP"] * 6
            ),
        ),
        (
            "tracks_csv",
            np.asarray(["AUS"] * 51 + ["SI"] * 48),
        ),
    ],
)
def test_basin(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.info.basin(data.lon, data.lat)

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "convention, expected",
    [
        (
            "Sainsbury2022JCLI",
            ["WEST"] + ["SUB", "MDR"] * 7 + [""] * 5,
        ),
        (
            "Sainsbury2022MWR",
            [""] * 3 + ["NoEurope", ""] * 6 + ["Europe", ""] * 2 + ["Europe"],
        ),
        (
            "Knutson2020",
            [
                "NATL",
                "ENP",
                "WNP",
                "NI",
                "SI",
                "SP",
                "SA",
            ],
        ),
    ],
)
def test_basin_definition(convention, expected):
    if convention == "Knutson2020":
        # Test specific coordinates for each Knutson2020 basin
        lon = np.array([-50, -120, 140, 70, 80, 160, -10])
        lat = np.array([30, 20, 25, 15, -20, -15, -25])
    else:
        lon = np.arange(-80, 20, 5)
        lat = np.asarray([15, 45] * 10)
    result = huracanpy.info.basin(lon, lat, convention=convention)

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            "tracks_minus180_plus180",
            np.asarray(
                [False]
                + [True] * 6
                + [False] * 2
                + [True] * 4
                + [False]
                + [True]
                + [False] * 6
                + [True] * 2
                + [True]
            ),
        ),
        (
            "tracks_0_360",
            np.asarray(
                [False]
                + [True] * 6
                + [False] * 2
                + [True] * 4
                + [False]
                + [True]
                + [False] * 6
                + [True] * 2
                + [True]
            ),
        ),
        ("tracks_csv", np.asarray([True] * 15 + [False] * 15 + [True] * 69)),
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
            np.asarray(
                [""]
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
            np.asarray(
                [""]
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
        ("tracks_csv", np.asarray([""] * 15 + ["Australia"] * 15 + [""] * 69)),
    ],
)
def test_get_country(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.info.country(data.lon, data.lat)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            "tracks_minus180_plus180",
            np.asarray(
                [""]
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
            np.asarray(
                [""]
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
        ("tracks_csv", np.asarray([""] * 15 + ["Oceania"] * 15 + [""] * 69)),
    ],
)
def test_get_continent(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.info.continent(data.lon, data.lat)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            "tracks_minus180_plus180",
            xr.open_dataset(str(data_path / "landfall_points_tracks_0_360_result.nc")),
        ),
        (
            "tracks_0_360",
            xr.open_dataset(str(data_path / "landfall_points_tracks_0_360_result.nc")),
        ),
        (
            "tracks_csv",
            xr.open_dataset(str(data_path / "landfall_points_tracks_csv_result.nc")),
        ),
    ],
)
def test_landfall_points(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.info.landfall_points(data.lon, data.lat, data.track_id)

    # Same result but different order for some reason
    xr.testing.assert_allclose(result.sortby("lon"), expected.sortby("lon"))
