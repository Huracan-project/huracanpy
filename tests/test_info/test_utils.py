import xarray as xr

import huracanpy


def test_inferred_track_id():
    tracks = huracanpy.load(huracanpy.example_csv_file)
    track_id = huracanpy.info.inferred_track_id(tracks.track_id)

    xr.testing.assert_equal(track_id, tracks.track_id)
