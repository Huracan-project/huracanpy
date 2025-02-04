import pytest

import numpy as np
import xarray as xr

import huracanpy


# %% DataArrayAccessor
def test_nunique():
    data = huracanpy.load(huracanpy.example_csv_file)

    N_tracks = data.track_id.hrcn.nunique()
    assert N_tracks == 3


@pytest.mark.parametrize("call_type", ["get", "add"])
@pytest.mark.parametrize(
    "function, function_args, accessor_name, accessor_function_kwargs",
    [
        (huracanpy.info.hemisphere, ["lat"], "hemisphere", {}),
        (huracanpy.info.basin, ["lon", "lat"], "basin", {}),
        (huracanpy.info.is_land, ["lon", "lat"], "is_land", {}),
        (huracanpy.info.is_ocean, ["lon", "lat"], "is_ocean", {}),
        (huracanpy.info.country, ["lon", "lat"], "country", {}),
        (huracanpy.info.continent, ["lon", "lat"], "continent", {}),
        (huracanpy.tc.ace, ["wind10"], "ace", {"wind_name": "wind10"}),
        (
            huracanpy.tc.ace,
            ["wind10", "track_id"],
            "ace",
            {"wind_name": "wind10", "sum_by": "track_id"},
        ),
        # (huracanpy.tc.pace, ["slp", "wind10"], "pace", {"pressure_name": "slp", "wind_name": "wind10"}),
        (huracanpy.info.season, ["track_id", "lat", "time"], "season", {}),
        (
            huracanpy.tc.saffir_simpson_category,
            ["wind10"],
            "saffir_simpson_category",
            {"wind_name": "wind10"},
        ),
        (
            huracanpy.tc.pressure_category,
            ["slp"],
            "pressure_category",
            {"slp_name": "slp"},
        ),
        (huracanpy.calc.distance, ["lon", "lat", "track_id"], "distance", {}),
        (
            huracanpy.calc.translation_speed,
            ["lon", "lat", "time", "track_id"],
            "translation_speed",
            {},
        ),
        (
            huracanpy.calc.delta,
            ["wind10", "track_id"],
            "delta",
            {"var_name": "wind10"},
        ),
        # (
        #     huracanpy.calc.get_rate,
        #     ["wind10", "time", "track_id"],
        #     "rate",
        #     {"var_name": "wind10"},
        # ),
        (
            huracanpy.calc.time_from_genesis,
            ["time", "track_id"],
            "time_from_genesis",
            {},
        ),
        (
            huracanpy.calc.time_from_apex,
            ["time", "track_id", "wind10"],
            "time_from_apex",
            {"intensity_var_name": "wind10"},
        ),
        (
            huracanpy.calc.track_duration,
            ["time", "track_id"],
            "track_duration",
            {},
        ),
        #        (huracanpy.calc.get_freq, ["track_id"], "freq", {}),
        #        (huracanpy.calc.get_tc_days, ["time", "track_id"], "tc_days", {}),
        # (huracanpy.calc.get_gen_vals, ["all", "time", "track_id"], "gen_vals", {}),
        # (
        #     huracanpy.calc.get_apex_vals,
        #     ["all", "wind10", "track_id"],
        #     "apex_vals",
        #     {"varname": "wind10"},
        # ),
    ],
)
def test_accessor_methods_match_functions(
    tracks_csv,
    function,
    function_args,
    accessor_name,
    accessor_function_kwargs,
    call_type,
):
    # Skip functions that only have a "get_" version
    if call_type == "add":
        if accessor_name in [
            "track_duration",
            "freq",
            "tc_days",
            "gen_vals",
            "apex_vals",
        ]:
            pytest.skip(f"Accessor function add_{accessor_name} does not exist")
        elif accessor_name in ["ace"] and "sum_by" in accessor_function_kwargs:
            pytest.skip(f"sum_by not a valid argument for add_{accessor_name}")

    # Call the huracanpy function
    # Get the function arguments as arrays. Use "all" as a wildcard for the full dataset
    function_args = [
        tracks_csv[var] if not var == "all" else tracks_csv for var in function_args
    ]
    result = function(*function_args)

    # Call the accessor method
    result_accessor = getattr(tracks_csv.hrcn, f"{call_type}_{accessor_name}")(
        **accessor_function_kwargs
    )

    # When using the "add_" method a new Dataset is returned with the variable added
    # The naming of the new variable is either simply the function name (minus "add_")
    # or the function name plus the name of the variable specified if it can be applied
    # to different variables
    if call_type == "add":
        varname = accessor_name
        if "var_name" in accessor_function_kwargs:
            varname = f"{varname}_{accessor_function_kwargs['var_name']}"
        result_accessor = result_accessor[varname]

    # Check that the function and method return identical results
    assert type(result) is type(
        result_accessor
    ), "accessor return type differs from function"
    np.testing.assert_equal(
        np.array(result),
        np.array(result_accessor),
        err_msg="accessor output differs from function output",
    )


# %% DatasetAccessor
# Currently keeping tests here that return more than just a DataArray as the testing is
# less generic
def test_get_methods(tracks_csv):
    """Test get_ accessors output is same as function"""
    data = tracks_csv

    ## - pace
    pace_acc, _ = data.hrcn.get_pace(pressure_name="slp", wind_name="wind10")
    pace_fct, model_fct = huracanpy.tc.pace(data.slp, data.wind10)
    np.testing.assert_array_equal(
        pace_acc, pace_fct, err_msg="accessor output differs from function output"
    )

    ## - time components
    year_acc, month_acc, day_acc, hour_acc = data.hrcn.get_time_components()
    year_fct, month_fct, day_fct, hour_fct = huracanpy.info.time_components(data.time)
    np.testing.assert_array_equal(
        year_acc, year_fct, err_msg="Year component does not match"
    )
    np.testing.assert_array_equal(
        month_acc, month_fct, err_msg="Month component does not match"
    )
    np.testing.assert_array_equal(
        day_acc, day_fct, err_msg="Day component does not match"
    )
    np.testing.assert_array_equal(
        hour_acc, hour_fct, err_msg="Hour component does not match"
    )

    ## - track pace
    pace_acc, _ = data.hrcn.get_pace(wind_name="wind10", sum_by="track_id")
    pace_fct, _ = huracanpy.tc.pace(data.slp, data.wind10, sum_by=data.track_id)
    np.testing.assert_array_equal(
        pace_acc,
        pace_fct,
        "Track PACE accessor output differs from function output",
    )

    ## - Genesis Values
    gen_vals_acc = data.hrcn.get_gen_vals(
        time_name="time",
        track_id_name="track_id",
    )
    gen_vals_fct = huracanpy.calc.gen_vals(data, data.time, data.track_id)
    xr.testing.assert_equal(gen_vals_acc, gen_vals_fct)

    ## - Apex Values
    apex_vals_acc = data.hrcn.get_apex_vals(
        track_id_name="track_id", varname="wind10", stat="max"
    )
    apex_vals_fct = huracanpy.calc.apex_vals(
        data, data.wind10, data.track_id, stat="max"
    )
    xr.testing.assert_equal(apex_vals_acc, apex_vals_fct)


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
