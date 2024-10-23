import huracanpy

import numpy as np


def test_get_methods():
    """Test get_ accessors output is same as function"""
    data = huracanpy.load(huracanpy.example_csv_file)

    ## - hemisphere
    hemi_acc = data.hrcn.get_hemisphere(lat_name="lat")
    hemi_fct = huracanpy.utils.get_hemisphere(data.lat)
    assert not any(hemi_acc != hemi_fct), "accessor output differs from function output"
    ## - basin
    basin_acc = data.hrcn.get_basin(lon_name="lon", lat_name="lat")
    basin_fct = huracanpy.utils.get_basin(data.lon, data.lat)
    assert not any(
        basin_acc != basin_fct
    ), "accessor output differs from function output"
    ## - land or ocean
    land_ocean_acc = data.hrcn.get_land_or_ocean(lon_name="lon", lat_name="lat")
    land_ocean_fct = huracanpy.utils.get_land_or_ocean(data.lon, data.lat)
    assert not any(
        land_ocean_acc != land_ocean_fct
    ), "accessor output differs from function output"
    ## - country
    country_acc = data.hrcn.get_country(lon_name="lon", lat_name="lat")
    country_fct = huracanpy.utils.get_country(data.lon, data.lat)
    assert not any(
        country_acc != country_fct
    ), "accessor output differs from function output"
    ## - continent
    continent_acc = data.hrcn.get_continent(lon_name="lon", lat_name="lat")
    continent_fct = huracanpy.utils.get_continent(data.lon, data.lat)
    assert not any(
        continent_acc != continent_fct
    ), "accessor output differs from function output"
    ## - ace
    ace_acc = data.hrcn.get_ace(wind_name="wind10")
    ace_fct = huracanpy.utils.get_ace(data.wind10)
    assert not any(ace_acc != ace_fct), "accessor output differs from function output"

    ## - pace
    pace_acc = data.hrcn.get_pace(pressure_name="slp", wind_name="wind10")
    pace_fct, model_fct = huracanpy.utils.get_pace(data.slp, data.wind10)
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

    ## - season
    season_acc = data.hrcn.get_season(
        track_id_name="track_id", lat_name="lat", time_name="time"
    )
    season_fct = huracanpy.utils.get_season(data.track_id, data.lat, data.time)
    assert all(season_acc == season_fct), "Season component does not match"

    ## - SSHS category
    sshs_acc = data.hrcn.get_sshs_cat(wind_name="wind10")
    sshs_fct = huracanpy.utils.get_sshs_cat(data.wind10)
    assert all(sshs_acc == sshs_fct), "SSHS category output does not match"

    ## - Pressure category
    pressure_cat_acc = data.hrcn.get_pressure_cat(slp_name="slp")
    pressure_cat_fct = huracanpy.utils.get_pressure_cat(data.slp)
    assert all(
        pressure_cat_acc == pressure_cat_fct
    ), "Pressure category output does not match"

    ## - Distance
    distance_acc = data.hrcn.get_distance(
        lon_name="lon", lat_name="lat", track_id_name="track_id"
    )
    distance_fct = huracanpy.utils.get_distance(data.lon, data.lat, data.track_id)
    np.testing.assert_array_equal(
        distance_acc,
        distance_fct,
        "Distance accessor output differs from function output",
    )

    ## - Translation speed
    translation_speed_acc = data.hrcn.get_translation_speed(
        lon_name="lon", lat_name="lat", time_name="time", track_id_name="track_id"
    )
    translation_speed_fct = huracanpy.utils.get_translation_speed(
        data.lon, data.lat, data.time, data.track_id
    )
    np.testing.assert_array_equal(
        translation_speed_acc,
        translation_speed_fct,
        "Translation speed  accessor output differs from function output",
    )

    ## - track duration
    duration_acc = data.hrcn.get_track_duration()
    duration_fct = huracanpy.diags.get_track_duration(data.time, data.track_id)
    np.testing.assert_array_equal(
        duration_acc,
        duration_fct,
        "duration accessor output differs from function output",
    )

    ## - track ace
    ace_acc = data.hrcn.get_track_ace(
        wind_name="wind10",
    )
    ace_fct = huracanpy.diags.get_track_ace(data.wind10, data.track_id)
    np.testing.assert_array_equal(
        ace_acc,
        ace_fct,
        "Track ACE accessor output differs from function output",
    )

    ## - track pace
    pace_acc = data.hrcn.get_track_pace(
        wind_name="wind10",
    )
    pace_fct = huracanpy.diags.get_track_pace(data.slp, data.track_id, data.wind10)
    np.testing.assert_array_equal(
        pace_acc,
        pace_fct,
        "Track PACE accessor output differs from function output",
    )

    ## - gen vals
    # gen_vals_acc = data.hrcn.get_gen_vals()
    # gen_vals_fct = huracanpy.diags.get_gen_vals(
    #    data,
    # )
    # TODO : Write test with same array function

    ## - apex vals
    # apex_vals_acc = data.hrcn.get_apexn_vals()
    # apex_vals_fct = huracanpy.diags.get_apex_vals(
    #    data,
    # )
    # TODO : Write test with same array function

    # TODO: lifecycle functions tests


def test_add_methods():
    """
    Test that add_ accessors output do add the columns

    """
    data = huracanpy.load(huracanpy.example_csv_file)

    data = (
        data.hrcn.add_hemisphere(lat_name="lat")
        .hrcn.add_basin(lon_name="lon", lat_name="lat")
        .hrcn.add_land_or_ocean(lon_name="lon", lat_name="lat")
        .hrcn.add_country(lon_name="lon", lat_name="lat")
        .hrcn.add_continent(lon_name="lon", lat_name="lat")
        .hrcn.add_ace(wind_name="wind10")
        .hrcn.add_pace(pressure_name="slp", wind_name="wind10")
        .hrcn.add_time_components(time_name="time")
        .hrcn.add_season(track_id_name="track_id", lat_name="lat", time_name="time")
        .hrcn.add_sshs_cat(wind_name="wind10")
        .hrcn.add_pressure_cat(slp_name="slp")
        .hrcn.add_distance(lon_name="lon", lat_name="lat")
        .hrcn.add_translation_speed(
            lon_name="lon", lat_name="lat", time_name="time", track_id_name="track_id"
        )
    )

    for col in [
        "hemisphere",
        "basin",
        "land_or_ocean",
        "country",
        "continent",
        "ace",
        "pace",
        "year",
        "month",
        "day",
        "hour",
        "season",
        "sshs_cat",
        "pressure_cat",
        "distance",
        "translation_speed",
    ]:
        assert col in list(data.variables), f"{col} not found in data columns"


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
