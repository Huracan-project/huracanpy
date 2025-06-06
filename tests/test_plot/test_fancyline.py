from cartopy.crs import EqualEarth
import matplotlib.pyplot as plt
import pytest

import huracanpy


@pytest.mark.parametrize("colors", [None, "wind10", "k"])
@pytest.mark.parametrize("linewidths", [None, "wind10", 3])
@pytest.mark.parametrize("alphas", [None, "wind10", 0.5])
@pytest.mark.parametrize(
    "linestyles", [None, lambda z: ["--" if value < 5 else "-" for value in z]]
)
@pytest.mark.parametrize("geoaxes", [True, False])
def test_fancyline(tracks_csv, colors, linewidths, alphas, linestyles, geoaxes):
    if colors in tracks_csv:
        colors = tracks_csv[colors]
    if linewidths in tracks_csv:
        linewidths = tracks_csv[linewidths]
    if alphas in tracks_csv:
        alphas = tracks_csv[alphas]
    if callable(linestyles):
        linestyles = linestyles(tracks_csv.wind10.values)
    if geoaxes:
        plt.axes(projection=EqualEarth())

    huracanpy.plot.fancyline(
        tracks_csv.lon,
        tracks_csv.lat,
        colors=colors,
        linewidths=linewidths,
        alphas=alphas,
        linestyles=linestyles,
    )
