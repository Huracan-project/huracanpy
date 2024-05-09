"""
Functions to plot track/genesis/whatever density

To compute the density, see huracanpy.diags.track_density
"""

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs

def plot_density(d, projection = ccrs.PlateCarree(180), cmap = "magma_r", levels=None):
    
    fig, ax = plt.subplots(subplot_kw = dict(projection = projection))
    ax.coastlines() 
    d.where(d > 0).plot(ax = ax, transform = ccrs.PlateCarree(), cmap = cmap, levels = levels)
    
    return fig, ax