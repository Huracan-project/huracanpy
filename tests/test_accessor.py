import pytest

import huracanpy

import numpy as np


# %% DataArrayAccessor
def test_nunique():
    data = huracanpy.load(huracanpy.example_csv_file)

    N_tracks = data.track_id.hrcn.nunique()
    assert N_tracks == 3


@pytest.mark.parametrize("call_type", ["get", "add"])
@pytest.mark.parametrize(
    "function, function_args, accessor_name, accessor_function_kwargs",
    [
        (huracanpy.utils.get_hemisphere, ["lat"], "hemisphere", {}),
        (huracanpy.utils.get_basin, ["lon", "lat"], "basin", {}),
        (huracanpy.utils.get_land_or_ocean, ["lon", "lat"], "land_or_ocean", {}),
        (huracanpy.utils.get_country, ["lon", "lat"], "country", {}),
        (huracanpy.utils.get_continent, ["lon", "lat"], "continent", {}),
        (huracanpy.tc.ace, ["wind10"], "ace", {"wind_name": "wind10"}),
        (
            huracanpy.tc.ace,
            ["wind10", "track_id"],
            "ace",
            {"wind_name": "wind10", "sum_by": "track_id"},
        ),
        # (huracanpy.tc.pace, ["slp", "wind10"], "pace", {"pressure_name": "slp", "wind_name": "wind10"}),
        (huracanpy.utils.get_season, ["track_id", "lat", "time"], "season", {}),
        (huracanpy.utils.get_sshs_cat, ["wind10"], "sshs_cat", {"wind_name": "wind10"}),
        (
            huracanpy.utils.get_pressure_cat,
            ["slp"],
            "pressure_cat",
            {"slp_name": "slp"},
        ),
        (huracanpy.utils.get_distance, ["lon", "lat", "track_id"], "distance", {}),
        (
            huracanpy.utils.get_translation_speed,
            ["lon", "lat", "time", "track_id"],
            "translation_speed",
            {},
        ),
        (
            huracanpy.utils.get_delta,
            ["wind10", "track_id"],
            "delta",
            {"var_name": "wind10"},
        ),
        (
            huracanpy.utils.get_rate,
            ["wind10", "time", "track_id"],
            "rate",
            {"var_name": "wind10"},
        ),
        (
            huracanpy.utils.get_time_from_genesis,
            ["time", "track_id"],
            "time_from_genesis",
            {},
        ),
        (
            huracanpy.utils.get_time_from_apex,
            ["time", "track_id", "wind10"],
            "time_from_apex",
            {"intensity_var_name": "wind10"},
        ),
        (
            huracanpy.diags.get_track_duration,
            ["time", "track_id"],
            "track_duration",
            {},
        ),
        (huracanpy.diags.get_freq, ["track_id"], "freq", {}),
        (huracanpy.diags.get_tc_days, ["time", "track_id"], "tc_days", {}),
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
        ]:
            pytest.skip(f"Accessor function add_{accessor_name} does not exist")
        elif accessor_name in ["ace"] and "sum_by" in accessor_function_kwargs:
            pytest.skip(f"sum_by not a valid argument for add_{accessor_name}")

    # Call the huracanpy function
    result = function(*[tracks_csv[var] for var in function_args])
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
def test_get_methods(tracks_csv):
    """Test get_ accessors output is same as function"""
    data = tracks_csv

    ## - pace
    pace_acc = data.hrcn.get_pace(pressure_name="slp", wind_name="wind10")
    pace_fct, model_fct = huracanpy.tc.pace(data.slp, data.wind10)
    assert not any(pace_acc != pace_fct), "accessor output differs from function output"

    ## - time components
    year_acc, month_acc, day_acc, hour_acc = data.hrcn.get_time_components(
        time_name="time"
    )
    year_fct, month_fct, day_fct, hour_fct = huracanpy.utils.get_time_components(
        data.time
    )
    assert all(year_acc == year_fct), "Year component does not match"
    assert all(month_acc == month_fct), "Month component does not match"
    assert all(day_acc == day_fct), "Day component does not match"
    assert all(hour_acc == hour_fct), "Hour component does not match"

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
    gen_vals_fct = huracanpy.diags.get_gen_vals(
        data,
    )
    assert gen_vals_acc.equals(
        gen_vals_fct
    ), "Genesis Values accessor output differs from function output"

    ## - Apex Values
    apex_vals_acc = data.hrcn.get_apex_vals(
        track_id_name="track_id", varname="wind10", stat="max"
    )
    apex_vals_fct = huracanpy.diags.get_apex_vals(data, varname="wind10", stat="max")
    assert apex_vals_acc.equals(
        apex_vals_fct
    ), "Genesis Values accessor output differs from function output"


def test_interp_methods():
    data = huracanpy.load(huracanpy.example_csv_file)
    interpolated_data_acc = data.hrcn.interp_time(
        freq="1h", track_id_name="track_id", prog_bar=False
    )
    expected_interpolated_data = huracanpy.utils.interp_time(
        data, freq="1h", track_id_name="track_id", prog_bar=False
    )
    np.testing.assert_array_equal(
        interpolated_data_acc.time, expected_interpolated_data.time
    )
