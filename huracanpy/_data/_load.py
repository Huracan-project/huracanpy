import pandas as pd

from . import _csv, _TRACK, _netcdf, _tempestextremes, _CHAZ, _MIT
from . import ibtracs
from huracanpy import utils


def load(
    filename=None,
    tracker=None,
    variable_names=None,
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

    The optional parameters for different trackers (currently **IBTrACS**, **TRACK** and
    **TempestExtremes**) are named {tracker}_{parameter} (in lower case),
    e.g. "ibtracs_online".

    Parameters
    ----------
    filename : str, optional
        The file to be loaded. If `tracker="ibtracs"`, this is not needed as the data is
        either included in huracanpy or downloaded when called
    tracker : str, optional
        If the file is not a CSV or NetCDF (identified by the file extension) then the
        tracker needs to be specified to decide how to load the data

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

    Returns
    -------
    xarray.Dataset

    """
    # If tracker is not given, try to derive the right function from the file extension
    if tracker is None:
        extension = filename.split(".")[-1]
        if extension == "csv":
            data = _csv.load(filename, **kwargs)
        elif extension == "parquet":
            data = _csv.load(filename, load_function=pd.read_parquet, **kwargs)
        elif filename.split(".")[-1] == "nc":
            data = _netcdf.load(filename, **kwargs)
        else:
            raise ValueError(f"{tracker} is set to None and file type is not detected")

    # If tracker is given, use the relevant function
    else:
        if tracker.lower() == "track":
            data = _TRACK.load(
                filename, calendar=track_calendar, variable_names=variable_names
            )
        elif tracker.lower() == "track.tilt":
            data = _TRACK.load_tilts(
                filename,
                calendar=track_calendar,
            )
        elif tracker.lower() in ["csv", "uz"]:
            data = _csv.load(filename)
        elif tracker.lower() in ["te", "tempest", "tempestextremes"]:
            data = _tempestextremes.load(
                filename,
                variable_names,
                tempest_extremes_unstructured,
                tempest_extremes_header_str,
            )
        elif tracker.lower() == "chaz":
            data = _CHAZ.load(filename)
        elif tracker.lower() == "mit":
            data = _MIT.load(filename, n_track_name, lat_track_name)
        elif tracker.lower() == "ibtracs":
            if ibtracs_online:
                if filename is None:
                    filename = "ibtracs.csv"

                with ibtracs.online(ibtracs_subset, filename, ibtracs_clean) as f:
                    data = _csv.load(
                        f,
                        read_csv_kws=dict(
                            header=0,
                            skiprows=[1],
                            na_values=["", " "],
                            keep_default_na=False,
                            converters={
                                "SID": str,
                                "SEASON": int,
                                "BASIN": str,
                                "SUBBASIN": str,
                                "LON": float,
                                "LAT": float,
                            },
                        ),
                    )
            else:
                data = _csv.load(ibtracs.offline(ibtracs_subset))
        else:
            raise ValueError(f"Tracker {tracker} unsupported or misspelled")

    if add_info:  # TODO : Manage potentially different variable names
        data["hemisphere"] = utils.geography.get_hemisphere(data.lat)
        data["basin"] = utils.geography.get_basin(data.lon, data.lat)
        data["season"] = utils.time.get_season(data.track_id, data.lat, data.time)
        if "wind10" in list(data.keys()):  # If 'wind10' is in the attributes
            data["sshs"] = utils.category.get_sshs_cat(data.wind10)
        if "slp" in list(data.keys()):  # If 'slp' is in the attributes
            data["pres_cat"] = utils.category.get_pressure_cat(data.slp)

    return data
