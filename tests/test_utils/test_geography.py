import huracanpy


def test_hemisphere():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    result = huracanpy.utils.geography.get_hemisphere(data.lat)

    assert (result == "S").all()


def test_basin():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    result = huracanpy.utils.geography.get_basin(data.lon, data.lat)

    assert (result[:51] == "AUS").all()
    assert (result[51:] == "SI").all()


def test_get_land_ocean():
    data = huracanpy.load(huracanpy.example_csv_file)
    result = huracanpy.utils.geography.get_land_or_ocean(data.lon, data.lat)

    assert (result[:15] == "Ocean").all()
    assert (result[15:30] == "Land").all()
    assert (result[30:] == "Ocean").all()


def test_get_country():
    data = huracanpy.load(huracanpy.example_csv_file)
    result = huracanpy.utils.geography.get_country(data.lon, data.lat)

    assert (result[:15] == "").all()
    assert (result[15:30] == "Australia").all()
    assert (result[30:] == "").all()
