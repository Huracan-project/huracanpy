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

    # Test that add_ accessors output do add the columns
    data = (
        data.hrcn.add_hemisphere(lat_name="lat")
        .hrcn.add_basin(lon_name="lon", lat_name="lat")
        .hrcn.add_land_or_ocean(lon_name="lon", lat_name="lat")
        .hrcn.add_country(lon_name="lon", lat_name="lat")
        .hrcn.add_continent(lon_name="lon", lat_name="lat")
    )

    for col in ["hemisphere", "basin", "land_or_ocean", "country", "continent"]:
        assert col in list(data.variables), f"{col} not found in data columns"
