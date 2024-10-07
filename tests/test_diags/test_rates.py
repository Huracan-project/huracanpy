import huracanpy


def test_translation_speed():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    deepening_rate = huracanpy.diags.rate(data, "slp")
    intensification_rate = huracanpy.diags.rate(data, "wind10")
    assert (
        deepening_rate.isel(mid_record=0).rate
        == (data.isel(record=1).slp - data.isel(record=0).slp) / 6
    )
    assert (
        intensification_rate.isel(mid_record=0).rate
        == (data.isel(record=1).wind10 - data.isel(record=0).wind10) / 6
    )
