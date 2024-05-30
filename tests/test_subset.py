import huracanpy


def test_trackswhere():
    tracks = huracanpy.load(huracanpy.example_csv_file)

    tracks["category"] = huracanpy.utils.category.get_pressure_cat(
        tracks.slp, slp_units="Pa"
    )

    tracks_subset = huracanpy.subset.trackswhere(
        tracks, lambda track: track.category.max() >= 2
    )

    assert set(tracks_subset.track_id.data) == {0, 2}
