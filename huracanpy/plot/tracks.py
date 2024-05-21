"""
Functions to plot the tracks
"""

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs


def plot_tracks_basic(
    tracks, intensity_var=None, projection=ccrs.PlateCarree(180), cmap="nipy_spectral"
):
    assert "lon" in list(tracks.keys()), "lon is not present in the data"
    assert "lat" in list(tracks.keys()), "lat is not present in the data"

    fig, ax = plt.subplots(subplot_kw=dict(projection=projection))
    ax.coastlines()
    sns.scatterplot(
        data=tracks, x="lon", y="lat", hue=intensity_var, ax=ax, s=2, palette=cmap
    )

    return fig, ax
