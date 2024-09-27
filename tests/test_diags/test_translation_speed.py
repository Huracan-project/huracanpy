import huracanpy


def test_translation_speed():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    v = huracanpy.diags.translation_speed.translation_speed(data)
    assert 6 <= v.translation_speed.mean() <= 6.1
    assert (
        len(v.mid_record)
        == len(data.time) - data.track_id.to_dataframe().nunique().values
    )[0]
