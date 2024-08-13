"""
Functions to plot track/genesis/whatever density

To compute the density, see huracanpy.diags.track_density
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def plot_density(
    d,
    contourf_kws=dict(cmap="magma_r", levels=10),
    subplot_kws=dict(projection=ccrs.PlateCarree(180)),
    fig_kws=dict(),
    cbar_kwargs={"label": ""},
):
    fig, ax = plt.subplots(subplot_kw=subplot_kws, **fig_kws)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    d.plot.contourf(
        ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs, **contourf_kws
    )

    return fig, ax
