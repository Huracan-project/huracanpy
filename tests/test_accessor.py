from inspect import getmembers, isfunction

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal
import numpy as np
import pytest
import xarray as xr

import huracanpy


# Functions not in the accessor
_intentionally_missing = [
    "load",
    "concat_tracks",
    # add_ functions that would have output with a different shape
    "add_apex_vals",
    "add_gen_vals",
    "add_density",
    "add_track_duration",
    "add_timestep",
    # plot_ functions that are for multiple datasets
    "plot_doughnut",
    "plot_venn",
]


def _get_function_args(tracks, function_args):
    # Get the function arguments as arrays. Use "all" as a wildcard for the full dataset
    # Keep other types of arguments as is
    return [
        var if not isinstance(var, str) else tracks if var == "all" else tracks[var]
        for var in function_args
    ]


# %% DataArrayAccessor
def test_nunique():
    data = huracanpy.load(huracanpy.example_csv_file)

    n_tracks = data.track_id.hrcn.nunique()
    assert n_tracks == 3


@pytest.mark.parametrize("call_type", ["get", "add"])
@pytest.mark.parametrize(
    "function, function_args, accessor_function_kwargs",
    [
        (huracanpy.info.hemisphere, ["lat"], {}),
        (huracanpy.info.basin, ["lon", "lat"], {}),
        (huracanpy.info.is_land, ["lon", "lat"], {}),
        (huracanpy.info.is_ocean, ["lon", "lat"], {}),
        (huracanpy.info.country, ["lon", "lat"], {}),
        (huracanpy.info.continent, ["lon", "lat"], {}),
        (
            huracanpy.info.category,
            ["wind10", [0, 10, 20, 30], [0, 1, 2]],
            {"var_name": "wind10", "bins": [0, 10, 20, 30], "labels": [0, 1, 2]},
        ),
        (huracanpy.info.season, ["track_id", "lat", "time"], {}),
        (huracanpy.info.timestep, ["time", "track_id"], {}),
        (huracanpy.info.time_components, ["time"], {}),
        (huracanpy.calc.density, ["lon", "lat"], {}),
        (huracanpy.calc.track_duration, ["time", "track_id"], {}),
        (huracanpy.calc.gen_vals, ["all", "time", "track_id"], {}),
        (
            huracanpy.calc.apex_vals,
            ["all", "wind10", "track_id"],
            {"var_name": "wind10"},
        ),
        (huracanpy.calc.time_from_genesis, ["time", "track_id"], {}),
        (
            huracanpy.calc.time_from_apex,
            ["time", "track_id", "wind10"],
            {"intensity_var_name": "wind10"},
        ),
        (huracanpy.calc.delta, ["wind10", "track_id"], {"var_name": "wind10"}),
        (
            huracanpy.calc.delta,
            ["wind10"],
            {"var_name": "wind10", "track_id_name": None},
        ),
        (huracanpy.calc.rate, ["wind10", "time", "track_id"], {"var_name": "wind10"}),
        (
            huracanpy.calc.rate,
            ["wind10", "time"],
            {"var_name": "wind10", "track_id_name": None},
        ),
        (huracanpy.calc.distance, ["lon", "lat", "track_id"], {}),
        (huracanpy.calc.distance, ["lon", "lat"], {"track_id_name": None}),
        (huracanpy.calc.azimuth, ["lon", "lat", "track_id"], {}),
        (huracanpy.calc.azimuth, ["lon", "lat"], {"track_id_name": None}),
        (huracanpy.calc.translation_speed, ["lon", "lat", "time", "track_id"], {}),
        (
            huracanpy.calc.translation_speed,
            ["lon", "lat", "time"],
            {"track_id_name": None},
        ),
        (huracanpy.tc.ace, ["wind10"], {"wind_name": "wind10"}),
        (
            huracanpy.tc.ace,
            ["wind10", "track_id"],
            {"wind_name": "wind10", "sum_by": "track_id"},
        ),
        (
            huracanpy.tc.pace,
            ["slp", "wind10"],
            {"pressure_name": "slp", "wind_name": "wind10"},
        ),
        (huracanpy.tc.saffir_simpson_category, ["wind10"], {"wind_name": "wind10"}),
        (
            huracanpy.tc.pressure_category,
            ["slp"],
            {"slp_name": "slp", "slp_units": "Pa"},
        ),
        # SLP for RMW is nonsense, but I'm just testing that the results are the same
        # and SLP is the best variable available for a reasonable order of magnitude
        (
            huracanpy.tc.beta_drift,
            ["lat", "wind10", "slp"],
            {"wind_name": "wind10", "rmw_name": "slp"},
        ),
    ],
)
def test_accessor_methods_match_functions(
    tracks_csv,
    function,
    function_args,
    accessor_function_kwargs,
    call_type,
):
    accessor_name = f"{call_type}_{function.__name__}"
    # Skip functions that only have a "get_" version
    if accessor_name in _intentionally_missing:
        pytest.skip(f"Accessor function {accessor_name} does not exist")
    if accessor_name == "add_ace" and "sum_by" in accessor_function_kwargs:
        pytest.skip(f"sum_by not a valid argument for {accessor_name}")

    # Call the huracanpy function
    function_args = _get_function_args(tracks_csv, function_args)
    if function == huracanpy.tc.pressure_category:
        with pytest.warns(UserWarning, match="Caution, pressure are likely in Pa"):
            result = function(*function_args)
    else:
        if (
            "track_id_name" in accessor_function_kwargs
            and accessor_function_kwargs["track_id_name"] is None
        ):
            with pytest.warns(UserWarning, match="track_id is not provided"):
                result = function(*function_args)
        else:
            result = function(*function_args)

    # Call the accessor method
    if (
        "track_id_name" in accessor_function_kwargs
        and accessor_function_kwargs["track_id_name"] is None
    ):
        with pytest.warns(UserWarning, match="track_id is not provided"):
            result_accessor = getattr(tracks_csv.hrcn, accessor_name)(
                **accessor_function_kwargs
            )
    else:
        result_accessor = getattr(tracks_csv.hrcn, accessor_name)(
            **accessor_function_kwargs
        )

    # Special case for PACE returning the values and the model
    if function == huracanpy.tc.pace:
        assert result[1] == result_accessor[1]
        result = result[0]
        result_accessor = result_accessor[0]

    # When using the "add_" method a new Dataset is returned with the variable added
    # The naming of the new variable is either simply the function name (minus "add_")
    # or the function name plus the name of the variable specified if it can be applied
    # to different variables
    if call_type == "add":
        # Special cases for multiple variables being added
        if accessor_name == "add_time_components":
            varname = ["year", "month", "day", "hour"]
            result = xr.Dataset({name: result[n] for n, name in enumerate(varname)})
        elif accessor_name == "add_beta_drift":
            varname = ["v_drift", "theta_drift"]
            result = xr.Dataset({name: result[n] for n, name in enumerate(varname)})
        else:
            varname = function.__name__
        if "var_name" in accessor_function_kwargs:
            varname = f"{varname}_{accessor_function_kwargs['var_name']}"
        result_accessor = result_accessor[varname]

    # Check that the function and method return identical results
    assert type(result) is type(result_accessor), (
        "accessor return type differs from function"
    )
    if isinstance(result, xr.Dataset):
        xr.testing.assert_identical(result, result_accessor)
    else:
        np.testing.assert_equal(
            np.asarray(result),
            np.asarray(result_accessor),
            err_msg="accessor output differs from function output",
        )


@check_figures_equal()
@pytest.mark.parametrize(
    "function, function_args, accessor_function_kwargs",
    [
        (huracanpy.plot.density, [], {}),
        (huracanpy.plot.fancyline, ["lon", "lat", "wind10"], {"colors": "wind10"}),
        (
            huracanpy.plot.tracks,
            ["lon", "lat", "wind10"],
            {"intensity_var_name": "wind10"},
        ),
        (huracanpy.plot.tracks, ["lon", "lat"], {}),
    ],
)
def test_accessor_plot_methods_match_functions(
    fig_test, fig_ref, tracks_csv, function, function_args, accessor_function_kwargs
):
    accessor_name = f"plot_{function.__name__}"

    if function is huracanpy.plot.density:
        function_args = [huracanpy.calc.density(tracks_csv.lon, tracks_csv.lat)]
    elif function is huracanpy.plot.fancyline:
        function_args = [
            _get_function_args(track, function_args)
            for track_id, track in tracks_csv.groupby("track_id")
        ]
    else:
        function_args = _get_function_args(tracks_csv, function_args)

    plt.figure(fig_ref)
    ax = plt.gca()

    if function is huracanpy.plot.fancyline:
        [function(*args, ax=ax) for args in function_args]
    elif (
        function is huracanpy.plot.tracks
        and "intensity_var_name" not in accessor_function_kwargs
    ):
        with pytest.warns(
            UserWarning, match="Ignoring `palette` because no `hue` variable"
        ):
            function(*function_args, ax=ax)
    else:
        function(*function_args, ax=ax)

    plt.figure(fig_test)
    ax = plt.gca()

    if (
        function is huracanpy.plot.tracks
        and "intensity_var_name" not in accessor_function_kwargs
    ):
        with pytest.warns(
            UserWarning, match="Ignoring `palette` because no `hue` variable"
        ):
            getattr(tracks_csv.hrcn, accessor_name)(**accessor_function_kwargs, ax=ax)
    else:
        getattr(tracks_csv.hrcn, accessor_name)(**accessor_function_kwargs, ax=ax)


def test_inferred_track_id(tracks_csv):
    track_id = tracks_csv.hrcn.get_inferred_track_id("track_id")
    xr.testing.assert_equal(track_id, tracks_csv.track_id)


def test_interp_methods():
    data = huracanpy.load(huracanpy.example_csv_file)
    interpolated_data_acc = data.hrcn.interp_time(
        freq="1h", track_id_name="track_id", prog_bar=False
    )
    expected_interpolated_data = huracanpy.interp_time(
        data, data.track_id, freq="1h", prog_bar=False
    )
    np.testing.assert_array_equal(
        interpolated_data_acc.time, expected_interpolated_data.time
    )


def test_accessor_sel_id(tracks_csv):
    result = tracks_csv.hrcn.sel_id(0)
    expected = huracanpy.sel_id(tracks_csv, tracks_csv.track_id, 0)

    xr.testing.assert_identical(result, expected)


def test_accessor_trackswhere(tracks_csv):
    result = tracks_csv.hrcn.trackswhere(lambda x: (x.track_id != 1).any())
    expected = huracanpy.trackswhere(
        tracks_csv, tracks_csv.track_id, lambda x: (x.track_id != 1).any()
    )

    xr.testing.assert_identical(result, expected)


def test_accessor_namespace_matches():
    # Functions at the top level have the same name as in the module
    expected_functions = [m[0] for m in getmembers(huracanpy) if isfunction(m[1])]

    # Functions in calc, info, and tc should have get_ and add_ equivalents in the
    # accessor
    expected_functions += [
        f"{prefix}_{m[0]}"
        for prefix in ["get", "add"]
        for module in [huracanpy.calc, huracanpy.info, huracanpy.tc]
        for m in getmembers(module)
        if isfunction(m[1])
    ]

    # Function in plot should be named plot_ in the accessor
    expected_functions += [
        f"plot_{m[0]}" for m in getmembers(huracanpy.plot) if isfunction(m[1])
    ]

    # Remove functions that do not go in the accessor
    for func in _intentionally_missing:
        expected_functions.remove(func)

    # The names of functions actually available on the accessor
    accessor_functions = [
        m[0]
        for m in getmembers(huracanpy._accessor.HuracanPyDatasetAccessor)
        if isfunction(m[1]) and m[0][0] != "_"
    ]

    if sorted(expected_functions) != sorted(accessor_functions):
        missing = [
            func for func in expected_functions if func not in accessor_functions
        ]
        extras = [func for func in accessor_functions if func not in expected_functions]

        raise ValueError(
            "Module and accessor functions do not match\n"
            + "Functions missing from accessor\n    - "
            + "\n    - ".join(missing)
            + "\nExtra functions in the accessor\n    - "
            + "\n    - ".join(extras)
        )
