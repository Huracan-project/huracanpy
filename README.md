# [![HuracanPy logo, a yellow and blue python spiralling as a cyclone.](docs/images/logo/logo-with-name/Slide1.png)](https://huracanpy.readthedocs.io/en/latest/)

[![Documentation Status](https://readthedocs.org/projects/huracanpy/badge/?version=latest)](https://huracanpy.readthedocs.io/en/latest/?badge=latest)
[![status](https://joss.theoj.org/papers/bb15b667a6306bcd0383d06d3b788cb6/status.svg)](https://joss.theoj.org/papers/bb15b667a6306bcd0383d06d3b788cb6)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/huracanpy/badges/version.svg)](https://anaconda.org/conda-forge/huracanpy)

*A python package for working with various forms of feature tracking data, including but not restricted to cyclone tracks.*

**Why HuracanPy?**
The idea of this package is to provide a unified tool for working with cyclone track data. 
In particular, HuracanPy can read tracks from many different sources/trackers. 
It also provides useful functions to analyse these tracks, including many common diagnostics.
Our goal is to make track data analysis more accessible, and to promote good reproducibility practices.


# Getting started
You can follow [user guide](https://huracanpy.readthedocs.io/en/latest/user_guide/index.html), try out some of the [examples](https://huracanpy.readthedocs.io/en/latest/examples/index.html), or follow the steps below (taken from the user guide).

## Install
You can install huracanpy with conda
```bash
conda install -c conda-forge huracanpy
```
or pip
```bash
python -m pip install huracanpy
```

To install the most up-to-date version of huracanpy, you can install directly from the repository with
```bash
python -m pip install "huracanpy@git+https://github.com/Huracan-project/huracanpy"
```

## Basic usage
The first step is to load in some tracks. HuracanPy can load track data from various sources as an [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) with a minimal number of assumed variables (track_id, lon, lat, time) e.g.
```python
import huracanpy

tracks = huracanpy.load(huracanpy.example_csv_file)
print(tracks)
```

```
<xarray.Dataset> Size: 7kB
Dimensions:   (record: 99)
Dimensions without coordinates: record
Data variables:
    track_id  (record) int64 792B 0 0 0 0 0 0 0 0 0 0 0 ... 2 2 2 2 2 2 2 2 2 2
    i         (record) int64 792B 482 476 476 477 478 ... 229 230 234 241 249
    j         (record) int64 792B 417 419 420 420 422 ... 501 509 517 528 542
    lon       (record) float64 792B 120.5 119.0 119.0 119.2 ... 58.5 60.25 62.25
    lat       (record) float64 792B -14.25 -14.75 -15.0 ... -39.25 -42.0 -45.5
    slp       (record) float64 792B 9.988e+04 9.981e+04 ... 9.747e+04 9.754e+04
    zs        (record) float64 792B -10.71 -16.11 -40.21 ... -218.5 -211.5
    wind10    (record) float64 792B 14.65 13.99 13.7 17.98 ... 23.69 23.96 23.4
    time      (record) datetime64[ns] 792B 1980-01-06T06:00:00 ... 1980-01-30...
```
Each "record" corresponds to a TC point (time, lon, lat).

Note that the data is one dimensional but represents multiple tracks.
This is done rather than having track_id as an additional dimension to avoid having to add blank data to each track when they are not the same length.
The `groupby` function, built in to xarray, allows us to easily loop over tracks in this format.
```python
# Iterating over all tracks
# Each track will be a subset of the xarray Dataset with a unique track_id
# The track_id is not necessarily an integer, it follows whatever you have loaded
# e.g. could be a string for IBTrACS
for track_id, track in tracks.groupby("track_id"):
    # Do something with the track
    print(track_id, len(track.time))
```
```
0 31
1 20
2 48
```

With the data loaded, we can apply the functions from HuracanPy. The example below is
using the `hrcn` accessor from HuracanPy. See the [accessor](accessor.ipynb) page for
more details.

```python
# Quickly view the tracks
tracks.hrcn.plot_tracks(intensity_var_name="wind10")

# Add a new variable to the tracks and plot this instead
tracks = tracks.hrcn.add_is_land()
tracks.hrcn.plot_tracks(intensity_var_name="is_land")
```
![Plot showing 3 tracks in the southern hemisphere with points coloured by wind speed](docs/images/readme/plot_tracks_wind.png)
![Plot showing 3 tracks in the southern hemisphere with points coloured by whether they are over land or ocean](docs/images/readme/plot_tracks_is_land.png)

# Contact
Please use GitHub's functions to communicate with HuracanPy's developers.
- Use [Issues](https://github.com/Huracan-project/huracanpy/issues) for feature requests or bug reporting
- Use the [Discussions](https://github.com/Huracan-project/huracanpy/discussions) for Q&A and general feedback 
- Do not forget HuracanPy is an open-source project, and you can also [contribute](https://huracanpy.readthedocs.io/en/latest/dev_guide/index.html) to it. 

## Subscribe for updates
1. Most specific: Subscribe to [this discussion](https://github.com/Huracan-project/huracanpy/discussions/57) for further updates.
2. Less specific: "Watch" the repo by clicking the button on the top-right of this page. Select "custom" then tick "discussions". You can always go back if there turns out to be too much emails. 
(We wish there was a better way for you to subscribe to announcements. If you agree with us, please up [this issue](https://github.com/orgs/community/discussions/3951).)
