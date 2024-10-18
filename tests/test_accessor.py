import huracanpy


def test_accessor():
    data = huracanpy.load(huracanpy.example_csv_file)

    # Test get_ accessors output is same as function
    ## - hemisphere
    hemi_acc = data.hrcn.get_hemisphere(lat_name="lat")
    hemi_fct = huracanpy.utils.geography.get_hemisphere(data.lat)
    assert not any(hemi_acc != hemi_fct), "accessor output differs from function output"
    ## - basin
    basin_acc = data.hrcn.get_basin(lon_name="lon", lat_name="lat")
    basin_fct = huracanpy.utils.geography.get_basin(data.lon, data.lat)
    assert not any(
        basin_acc != basin_fct
    ), "accessor output differs from function output"
    ## - land or ocean
    land_ocean_acc = data.hrcn.get_land_or_ocean(lon_name="lon", lat_name="lat")
    land_ocean_fct = huracanpy.utils.geography.get_land_or_ocean(data.lon, data.lat)
    assert not any(
        land_ocean_acc != land_ocean_fct
    ), "accessor output differs from function output"
    ## - country
    country_acc = data.hrcn.get_country(lon_name="lon", lat_name="lat")
    country_fct = huracanpy.utils.geography.get_country(data.lon, data.lat)
    assert not any(
        country_acc != country_fct
    ), "accessor output differs from function output"
    ## - continent
    continent_acc = data.hrcn.get_continent(lon_name="lon", lat_name="lat")
    continent_fct = huracanpy.utils.geography.get_continent(data.lon, data.lat)
    assert not any(
        continent_acc != continent_fct
    ), "accessor output differs from function output"
    ## - ace
    ace_acc = data.hrcn.get_ace(wind_name="wind")
    ace_fct = huracanpy.utils.ace.get_ace(data.wind)
    assert not any(ace_acc != ace_fct), "accessor output differs from function output"

    ## - pace
    pace_acc, model_acc = data.hrcn.get_pace(pressure_name="pressure", wind_name="wind")
    pace_fct, model_fct = huracanpy.utils.ace.get_pace(data.pressure, data.wind)
    assert not any(pace_acc != pace_fct), "accessor output differs from function output"

    ## - time components
    year_acc, month_acc, day_acc, hour_acc = data.hrcn.get_time_components(
        time_name="time"
    )
    year_fct, month_fct, day_fct, hour_fct = huracanpy.utils.time.get_time_components(
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
    season_fct = huracanpy.utils.time.get_season(data.track_id, data.lat, data.time)
    assert all(season_acc == season_fct), "Season component does not match"

    # Test that add_ accessors output do add the columns
    data = (
        data.hrcn.add_hemisphere(lat_name="lat")
        .hrcn.add_basin(lon_name="lon", lat_name="lat")
        .hrcn.add_land_or_ocean(lon_name="lon", lat_name="lat")
        .hrcn.add_country(lon_name="lon", lat_name="lat")
        .hrcn.add_continent(lon_name="lon", lat_name="lat")
        .hrcn.add_ace(wind_name="wind")
        .hrcn.add_pace(pressure_name="pressure", wind_name="wind")
        .hrcn.add_time_components(time_name="time")
        .hrcn.add_season(track_id_name="track_id", lat_name="lat", time_name="time")
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
    ]:
        assert col in list(data.variables), f"{col} not found in data columns"
