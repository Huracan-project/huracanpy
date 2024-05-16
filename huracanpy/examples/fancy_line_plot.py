import matplotlib.pyplot as plt
from cartopy.crs import EqualEarth

import huracanpy
from huracanpy.plot import fancyline


# Load in a single track from example data
tracks = huracanpy.load(huracanpy.example_csv_file)
track = tracks.groupby("track_id")[0]

# Set up a figure with a cartopy projection
fig, axes = plt.subplots(
    2,
    2,
    sharex="all",
    sharey="all",
    figsize=(12, 8),
    subplot_kw=dict(projection=EqualEarth()),
)

# Show 10m wind speed with a colourscale
lc = fancyline(track.lon, track.lat, track.wind10, vmin=10, vmax=25, ax=axes[0, 0])
plt.colorbar(lc, extend="both")

# Show 10m wind speed with linewidth
fancyline(
    track.lon,
    track.lat,
    linewidths=track.wind10,
    wmin=10,
    wmax=25,
    wrange=(1, 10),
    ax=axes[0, 1],
)

# Show 10m wind speed with alpha (transparency)
# Example with other arguments as single values for all lines
fancyline(
    track.lon,
    track.lat,
    alphas=track.wind10,
    amin=10,
    amax=25,
    arange=(0.5, 1),
    colors="k",
    linewidths=3,
    ax=axes[1, 0]
)

# Use the linestyle as categorical whether 10m wind is greater than a threshold
linestyles = ["--" if x < 20 else "-" for x in track.wind10]
fancyline(track.lon, track.lat, linestyles=linestyles, ax=axes[1, 1])

# Make all axes look nice
for ax in axes.flatten():
    ax.coastlines()
    ax.gridlines(draw_labels=True)

plt.show()
