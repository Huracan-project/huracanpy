import pandas as pd

from . import _csv, _TRACK, _netcdf, _tempestextremes, _CHAZ, _MIT
from . import ibtracs
from .. import utils


rename_defaults = dict(
    longitude="lon",
    latitude="lat",
)


def load(
    filename=None,
    source=None,
    variable_names=None,
    rename=None,
    add_info=False,
    ibtracs_online=False,
    ibtracs_subset="wmo",
    ibtracs_clean=True,
    tempest_extremes_unstructured=False,
    tempest_extremes_header_str="start",
    track_calendar=None,
    n_track_name="n_track",
    lat_track_name="lat_track",
    **kwargs,
):
    """Load track data

    The optional parameters for different sources of tracks (currently **IBTrACS**,
    **TRACK** and **TempestExtremes**) are named {source}_{parameter} (in lower case),
    e.g. "ibtracs_online".

    Parameters
    ----------
    filename : str, optional
        The file to be loaded. If `source="ibtracs"`, this is not needed as the data is
        either included in huracanpy or downloaded when called
    source : str, optional
        If the file is not a CSV or NetCDF (identified by the file extension) then the
        source needs to be specified to decide how to load the data

        * **track**
        * **track.tilt**
        * **te**, **tempest**, **tempestextremes**, **uz**:
        * **ibtracs**
        * **CHAZ**, **MIT**

    variable_names : list of str, optional
          When loading data from an ASCII file (TRACK or TempestExtremes), specify the
          list of variables that have been added to the tracks. This does not include
          variables which are in the track by default (e.g. time/lon/lat/vorticity for
          TRACK). If a variable at multiple levels has been added to the tracks, then a
          variable name for each level needs to be included (e.g. vorticity profiles
          with TRACK). If the variable names are not given, then any additional
          variables will be named variable_n, where n goes from 0 to the number of
          variables

    rename : dict, optional
        A mapping of variable names to rename. Defaults are

        * longitude -> lon
        * latitude -> lat

        To override any of these defaults, you can pass the same name, e.g.

        >>> tracks = huracanpy.load(..., rename=dict(longitude="longitude"))

    add_info : bool, default=False
    ibtracs_online : bool, default=False
        * **False**: Use a small subset of the IBTrACS data included in this package
        * **True**: Download the IBTrACS data
    ibtracs_subset : str, default="ALL"
        IBTrACS subset. When loading offline data it is one of

        * **WMO**: Data with the wmo_* variables. The data as reported by the WMO agency
          responsible for each basin, so methods are not consistent across basins
        * **USA** or **JTWC**: Data with the usa_* variables. The data as recorded by
          the USA/Joint Typhoon Warning Centre. Methods are consistent across basins,
          but may not be complete.

        If you are downloading the online data, the subsets are the different files
        provided by IBTrACS

        * **ACTIVE**: TCs currently active
        * **ALL**: Entire IBTrACS database
        * Specific basins: **EP**, **NA**, **NI**, **SA**, **SI**, **SP**, **WP**
        * **last3years**: self-explanatory
        * **since1980**: Entire IBTrACS database since 1980 (advent of satellite era,
          considered reliable from then on)

    ibtracs_clean : bool, default=True
        If downloading IBTrACS data, this parameter says whether to delete the
        downloaded file after loading it into memory.

    tempest_extremes_unstructured : bool, default=False,
        By default the first two columns in TempestExtremes files are the i, j indices
        of the closest gridpoint, but for unstructured grids it is a single lookup index
        so there is one less column
    tempest_extremes_header_str : str, default="start"
        This is an option in the Colin's load function, so I assume this can change
        between files

    track_calendar : str, optional
          When loading data from a TRACK ASCII file, if the data uses a different
          calendar to the default :class:`datetime.datetime`, then you can pass this
          argument to load the times in as :class:`cftime.datetime` with the given
          calendar instead
    **kwargs
        When loading tracks from a standard files these will be passed to the relevant
        load function

        * netCDF file - :func:`xarray.open_dataset`
        * CSV file - :func:`pandas.read_csv`
        * parquet file - :func:`pandas.read_parquet`

        For CSV files pandas interprets "NA" as `nan` by default, which is overriden in
        this function. To restore the pandas default behavious set
        :code:`keep_default_NA=True` and :code:`na_values=[]`

    Returns
    -------
    xarray.Dataset

    """
    # If source is not given, try to derive the right function from the file extension
    if source is None:
        extension = filename.split(".")[-1]
        if extension == "csv":
            data = _csv.load(filename, **kwargs)
        elif extension == "parquet":
            data = _csv.load(filename, load_function=pd.read_parquet, **kwargs)
        elif filename.split(".")[-1] == "nc":
            data = _netcdf.load(filename, **kwargs)
        else:
            raise ValueError(f"{source} is set to None and file type is not detected")

    # If source is given, use the relevant function
    else:
        if source.lower() == "track":
            data = _TRACK.load(
                filename, calendar=track_calendar, variable_names=variable_names
            )
        elif source.lower() == "track.tilt":
            data = _TRACK.load_tilts(
                filename,
                calendar=track_calendar,
            )
        elif source.lower() in ["csv", "uz"]:
            data = _csv.load(filename, **kwargs)
        elif source.lower() in ["te", "tempest", "tempestextremes"]:
            data = _tempestextremes.load(
                filename,
                variable_names,
                tempest_extremes_unstructured,
                tempest_extremes_header_str,
            )
        elif source.lower() == "chaz":
            data = _CHAZ.load(filename)
        elif source.lower() == "mit":
            data = _MIT.load(filename, n_track_name, lat_track_name)
        elif source.lower() == "ibtracs":
            if ibtracs_online:
                if filename is None:
                    filename = "ibtracs.csv"

                with ibtracs.online(ibtracs_subset, filename, ibtracs_clean) as f:
                    # Put IBTrACS specific arguments to read_csv second, so it
                    # overwrites any arguments passed
                    kwargs = {
                        **kwargs,
                        **dict(
                            header=0,
                            skiprows=[1],
                            converters={
                                "SID": str,
                                "SEASON": int,
                                "BASIN": str,
                                "SUBBASIN": str,
                                "LON": float,
                                "LAT": float,
                            },
                        ),
                    }
                    return load(
                        filename=f,
                        source="csv",
                        rename=rename,
                        add_info=add_info,
                        **kwargs,
                    )
            else:
                return load(
                    filename=ibtracs.offline(ibtracs_subset),
                    rename=rename,
                    add_info=add_info,
                    **kwargs,
                )
        else:
            raise ValueError(f"Source {source} unsupported or misspelled")

    if rename is None:
        rename = rename_defaults
    else:
        # Overwrite default arguments with explicit arguments passed to rename by
        # putting "rename" second in this dictionary combination
        rename = {**rename_defaults, **rename}

    # xarray.Dataset.rename only accepts keys that are actually in the dataset
    rename = {key: rename[key] for key in rename if key in data}

    if len(rename) > 0:
        data = data.rename(rename)

    if add_info:  # TODO : Manage potentially different variable names
        data["hemisphere"] = utils.geography.get_hemisphere(data.lat)
        data["basin"] = utils.geography.get_basin(data.lon, data.lat)
        data["season"] = utils.time.get_season(data.track_id, data.lat, data.time)
        if "wind10" in list(data.keys()):  # If 'wind10' is in the attributes
            data["sshs"] = utils.category.get_sshs_cat(data.wind10)
        if "slp" in list(data.keys()):  # If 'slp' is in the attributes
            data["pres_cat"] = utils.category.get_pressure_cat(data.slp)

    return data
