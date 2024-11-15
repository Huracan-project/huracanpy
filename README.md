# huracanpy
[![Documentation Status](https://readthedocs.org/projects/huracanpy/badge/?version=latest)](https://huracanpy.readthedocs.io/en/latest/?badge=latest)

A python package for working with various forms of feature tracking data

## Version 1 coming soon! Subscribe for notification.
We will release version 1 soon (by the end of November 2024). We advice you wait for this version to install the package. To stay updated about it, there are two options:
1. Most specific: Subscribe to [this discussion](https://github.com/Huracan-project/huracanpy/discussions/57), that we will use to communicate with our community of users.
2. Less specific: "Watch" the repo by clicking the button on the top-right of this page. Select "custom" then tick "discussions". You can always go back if there turns out to be too much emails. 
(We wish there was a better way for you to subscribe to announcements. If you agree with us, please up [this issue](https://github.com/orgs/community/discussions/3951).)

## Installation
To install the package, you can use `pip`: `pip install huracanpy`.

This can fail with older python versions due to issues with installing cartopy through
pip. If this happens, use conda to install cartopy first
(e.g. `conda install -c conda-forge cartopy`), then install huracanpy as normal


## Usage
The idea of this package is to be a standard way for working with cyclone track data. We
were all working on track data, but in slightly different ways which makes sharing code
more difficult. The method chosen here is to treat a set of tracks as an xarray
[Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) with a
minimal number of assumed variables (track_id, lon, lat, time). e.g. running

```python
import huracanpy

tracks = huracanpy.load(huracanpy.example_csv_file)
minimal_tracks = tracks[["track_id", "lon", "lat", "time"]]

print(minimal_tracks)
```
gives
```
<xarray.Dataset>
Dimensions:   (obs: 99)
Coordinates:
  * obs       (obs) int64 0 1 2 3 4 5 6 7 8 9 ... 89 90 91 92 93 94 95 96 97 98
Data variables:
    track_id  (obs) int64 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 2 2 2 2 2 2 2 2 2 2 2 2
    lon       (obs) float64 120.5 119.0 119.0 119.2 ... 57.5 58.5 60.25 62.25
    lat       (obs) float64 -14.25 -14.75 -15.0 -15.0 ... -39.25 -42.0 -45.5
    time      (obs) datetime64[ns] 1980-01-06T06:00:00 ... 1980-01-30T18:00:00
```

where each "obs" corresponds to a TC point (time, lon, lat).
All variables that were present in your file are variables in the loaded dataset.

Note that the data is one dimensional but represents multiple tracks. This is done
rather than having track_id as an additional dimension to avoid having to add a bunch of
extra blank data to each track when they are not the same length. The `groupby` function
allows us to easily loop over or index tracks in this format.
```python
import huracanpy

tracks = huracanpy.load(huracanpy.example_csv_file)

track_groups = tracks.groupby("track_id")

# Selecting track by ID
# The track_id is not necessarily an integer, it follows whatever you have loaded
# e.g. could be a string for IBTrACS
track_id1 = track_groups[1]

# Iterating over all tracks
# Each track will be a subset of the xarray Dataset with a unique track_id
for n, track in track_groups:
    # Do something with the track
```

#### csv tracks data
If you tracks are stored in csv (including if they were outputed from TempestExtremes' StitchNodes),
you can specify the `tracker="csv"` argument, or, if your filename ends with `csv`, it will be detected automatically.

#### TRACK tracks data
If your tracks are in TRACK format, use the `tracker="TRACK"` option

## Contributing
We welcome suggestions and contributions to huracanpy. If you have existing code for
working with track data, we would definitely welcome contributions/suggestions for
modifying it to work with the xarray format used here. To add suggestions, feel free to
open an issue or contact us. To contribute code, please make a fork of this repository,
follow the instructions below, and open a pull request.

To install your copy locally run
```shell
pip install -e .[dev]
```
The "[dev]" argument installs the following optional packages that are useful for
contributing to development
1. **pytest**

    We use [pytest](https://docs.pytest.org/en/latest/) to run automated tests. If you
    add a new feature, it would be good to also add tests to check that feature is
    working and keeps working in the future. You can also run `pytest` from the top
    level directory of the package to check that your changes haven't broken anything.
2. **ruff**

    We use [ruff](https://docs.astral.sh/ruff/) to automatically keep the style of the
    code consistent so we don't have to worry about it. To check that your code passes
    you can run `ruff check` and `ruff format --check`. To automatically fix differences
    run `ruff check --fix` and `ruff format`.

3. **pre-commit**

    You can use [pre-commit](https://pre-commit.com/) to automate the formatting done by
    ruff. After running `pre-commit install` at the top level directory, any future git
    commits will automatically run the ruff formatting on any files you have changes.
