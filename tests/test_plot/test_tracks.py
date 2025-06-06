import huracanpy


def test_plot_tracks(tracks_csv):
    huracanpy.plot.tracks(tracks_csv.lon, tracks_csv.lat, tracks_csv.wind10)
