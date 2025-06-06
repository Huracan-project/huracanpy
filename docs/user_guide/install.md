# Installation

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

## Dependencies
The dependencies are managed automatically by the installation. For interest, below are
the  dependencies for huracanpy as listed in the `pyproject.toml` and descriptions of
what they are used for in HuracanPy.

```toml
dependencies = [
    "numpy",
    "xarray",
    "cftime",
    "parse",
    "shapely",
    "pandas",
    # Issue with newer versions of fiona when using old versions of geopandas
    # see https://stackoverflow.com/a/78949565/8270394
    "geopandas>=0.14.4",
    "matplotlib",
    "seaborn",
    "netcdf4",
    "haversine",
    "cartopy",
    "matplotlib-venn<1",
    "metpy",
    "tqdm",
    "pyarrow",
    "pyproj"
]
```

### Loading data
- [**xarray**](https://docs.xarray.dev/en/stable/) (with [**netcdf4**](https://unidata.github.io/netcdf4-python/)) is used for loading netCDF files as well as providing the Dataset object that all file types are loaded into
- [**pandas**](https://pandas.pydata.org/docs/) (with [**pyarrow**](https://arrow.apache.org/docs/python/index.html)) loads CSV files (and text files that can be reformatted as CSV). **pyarrow** adds support for .parquet files with **pandas**
- [**numpy**](https://numpy.org/doc/stable/) is used for loading other text files
- [**parse**](https://github.com/r1chardj0n3s/parse) is used to extract data from specifically formatted lines in text files
- [**cftime**](https://unidata.github.io/cftime/) adds support for non-standard calendars (including with **xarray**)

### Analysis
- [**xarray**](https://docs.xarray.dev/en/stable/) provides the Dataset object that HuracanPy is largely built around
- [**pandas**](https://pandas.pydata.org/docs/) is used where it can do the equivalent of **xarray** but much faster
- [**numpy**](https://numpy.org/doc/stable/) is underpinning **xarray** and many functions in HuracanPy can be used with pure **numpy** arrays
- [**metpy**](https://unidata.github.io/MetPy/latest/index.html) gives us support for unit-aware calculations in a number of functions

### Geospatial
- [**geopandas**](https://geopandas.org/en/stable/) (with [**shapely**](https://shapely.readthedocs.io/en/stable/index.html) and [**cartopy**](https://scitools.org.uk/cartopy/docs/latest/)) is used to match track points with Earth features (land, sea, country, etc.), with **shapely** used to interface with **geopandas** and **cartopy** providing a useful method for downloading and caching feature files 
- [**pyproj**](https://pyproj4.github.io/pyproj/stable/) and [**haversine**](https://github.com/mapado/haversine) provide the distance and angle calculations

### Plotting
- [**matplotlib**](https://matplotlib.org/stable/) is the basis for any plotting functions
- [**seaborn**](https://seaborn.pydata.org/) is used for `huracanpy.plot.tracks`
- [**cartopy**](https://scitools.org.uk/cartopy/docs/latest/) adds support for projections and transforms on plots
- [**matplotlib-venn**](https://github.com/konstantint/matplotlib-venn) is used for `huracanpy.plot.venn`

### Other
- [**tqdm**](https://tqdm.github.io/) adds an optional progress bar to `huracanpy.interp_time`
