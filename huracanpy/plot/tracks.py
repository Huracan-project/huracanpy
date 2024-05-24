"""
Functions to plot the tracks
"""

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs


def plot_tracks_basic(
    tracks,
    intensity_var=None,
    subplot_kws=dict(projection=ccrs.PlateCarree(180)),
    fig_kws=dict(figsize=(10, 10)),
    scatter_kws=dict(palette="nipy_spectral", s=2),
):
    assert "lon" in list(tracks.keys()), "lon is not present in the data"
    assert "lat" in list(tracks.keys()), "lat is not present in the data"

    fig, ax = plt.subplots(subplot_kw=subplot_kws, **fig_kws)
    ax.coastlines()
    sns.scatterplot(
        data=tracks, x="lon", y="lat", hue=intensity_var, ax=ax, **scatter_kws
    )

    return fig, ax
