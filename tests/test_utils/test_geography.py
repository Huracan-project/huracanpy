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
