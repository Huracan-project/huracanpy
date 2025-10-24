import huracanpy


def test_plot_density(tracks_year):
    density = huracanpy.calc.density(tracks_year.lon, tracks_year.lat, spherical=True)
    huracanpy.plot.density(density)
