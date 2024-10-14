"""
Functions to plot the tracks
"""

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs


def tracks(
    lon,
    lat,
    intensity_var=None,
    subplot_kws=dict(projection=ccrs.PlateCarree(180)),
    fig_kws=dict(figsize=(10, 10)),
    scatter_kws=dict(palette="nipy_spectral", s=2, color="k"),
):
    fig, ax = plt.subplots(subplot_kw=subplot_kws, **fig_kws)
    ax.coastlines()
    sns.scatterplot(
        x=lon,
        y=lat,
        hue=intensity_var,
        ax=ax,
        **scatter_kws,
        transform=ccrs.PlateCarree(),
    )

    return fig, ax
