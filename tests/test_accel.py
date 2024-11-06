"""
Test functions that use tricks to speed up their code produce the same result as the
slower method
"""

from haversine import haversine_vector, Unit
import numpy as np
import xarray as xr

import huracanpy


def test_accel_sel_id(tracks_csv):
    result = huracanpy.sel_id(tracks_csv, tracks_csv.track_id, 0)

    expected = tracks_csv.groupby("track_id")[0]

    xr.testing.assert_identical(result, expected)


def test_accel_trackswhere():
    # TODO accelerate trackswhere
    pass


def test_accel_get_gen_vals(tracks_csv):
    result = huracanpy.calc.get_gen_vals(
        tracks_csv, tracks_csv.time, tracks_csv.track_id
    )

    expected = tracks_csv.groupby("track_id").first()

    xr.testing.assert_identical(result, expected)


def test_accel_get_apex_vals(tracks_csv):
    result = huracanpy.calc.get_apex_vals(
        tracks_csv, tracks_csv.wind10, tracks_csv.track_id
    )

    expected = tracks_csv.sortby("wind10", ascending=False).groupby("track_id").first()

    xr.testing.assert_identical(result, expected)


def test_accel_get_time_from_genesis(tracks_csv):
    result = huracanpy.calc.get_time_from_genesis(tracks_csv.time, tracks_csv.track_id)

    track_groups = tracks_csv.groupby("track_id")
    expected = []
    for track_id, track in track_groups:
        expected.append(track.time - track.time[0])

    expected = xr.concat(expected, dim="record")
    expected = expected.rename("time_from_genesis")

    xr.testing.assert_identical(result, expected)


def test_accel_get_time_from_apex(tracks_csv):
    result = huracanpy.calc.get_time_from_apex(
        tracks_csv.time, tracks_csv.track_id, tracks_csv.wind10
    )

    track_groups = tracks_csv.groupby("track_id")
    expected = []
    for track_id, track in track_groups:
        idx = track.wind10.argmax()
        expected.append(track.time - track.time[idx])

    expected = xr.concat(expected, dim="record")
    expected = expected.rename("time_from_extremum")

    xr.testing.assert_identical(result, expected)


def test_accel_match():
    ref = huracanpy.load(huracanpy.example_csv_file)
    tracks = ref.where(ref.track_id < 2, drop=True)
    tracks = tracks.where(tracks.time.dt.hour == 0, drop=True)
    tracks["lon"] = tracks.lon + 0.5
    tracks["lat"] = tracks.lat + 0.5

    result = huracanpy.assess.match([tracks, ref])

    max_dist = 300
    track_id1 = []
    track_id2 = []
    npoints = []
    dist = []

    for track_id, track in tracks.groupby("track_id"):
        for track_id_ref, track_ref in ref.groupby("track_id"):
            # Match times
            track_ = track.where(track.time.isin(track_ref.time), drop=True)

            if len(track_.time) > 0:
                track_ref_ = track_ref.where(track_ref.time.isin(track.time), drop=True)

                yx_track = np.array([track_.lat, track_.lon]).T
                yx_ref = np.array([track_ref_.lat, track_ref_.lon]).T

                dists = haversine_vector(yx_track, yx_ref, Unit.KILOMETERS)

                matches = dists < max_dist
                if matches.any():
                    track_id1.append(track_id)
                    track_id2.append(track_id_ref)

                    dists_track = dists[matches]
                    npoints.append(len(dists_track))
                    dist.append(np.mean(dists_track))

    np.testing.assert_equal(result.id_1, np.array(track_id1))
    np.testing.assert_equal(result.id_2, np.array(track_id2))
    np.testing.assert_equal(result.temp, np.array(npoints))
    np.testing.assert_allclose(result.dist, np.array(dist), rtol=1e-12)


def test_accel_overlap():
    ref = huracanpy.load(huracanpy.example_csv_file)
    tracks = ref.where(ref.track_id < 2, drop=True)
    tracks = tracks.where(tracks.time.dt.hour == 0, drop=True)
    tracks["lon"] = tracks.lon + 0.5
    tracks["lat"] = tracks.lat + 0.5

    result = huracanpy.assess.overlap(tracks, ref)

    delta_start = []
    delta_end = []

    for n, row in result.iterrows():
        track = tracks.where(tracks.track_id == row.id_1, drop=True)
        track_ref = ref.where(ref.track_id == row.id_2, drop=True)

        delta_start.append((track_ref.time[0] - track.time[0]) / np.timedelta64(1, "D"))
        delta_end.append((track_ref.time[-1] - track.time[-1]) / np.timedelta64(1, "D"))

    np.testing.assert_equal(result.delta_start, np.array(delta_start))
    np.testing.assert_equal(result.delta_end, np.array(delta_end))
