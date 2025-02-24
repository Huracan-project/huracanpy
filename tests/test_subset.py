import numpy as np

import huracanpy


def test_sel_id(tracks_csv):
    # Check that only the points with the track id are extracted each time and all of
    # the points are extracted once
    npoints = 0
    for track_id in np.unique(tracks_csv.track_id):
        tracks_subset = huracanpy.sel_id(tracks_csv, tracks_csv.track_id, track_id)
        np.testing.assert_array_equal(tracks_subset.track_id, track_id)
        npoints += len(tracks_subset.record)

    assert npoints == len(tracks_csv.record)


def test_sel_id_array(tracks_csv):
    npoints = 0
    for track_id in [[0, 1], [2]]:
        tracks_subset = huracanpy.sel_id(tracks_csv, tracks_csv.track_id, track_id)
        assert np.isin(tracks_subset.track_id, track_id).all()
        npoints += len(tracks_subset.record)

    assert npoints == len(tracks_csv.record)


def test_trackswhere():
    tracks = huracanpy.load(huracanpy.example_csv_file)

    tracks["category"] = huracanpy.tc.pressure_category(tracks.slp, slp_units="Pa")

    tracks_subset = huracanpy.trackswhere(
        tracks, tracks.track_id, lambda track: track.category.max() >= 2
    )

    assert set(tracks_subset.track_id.data) == {0, 2}
