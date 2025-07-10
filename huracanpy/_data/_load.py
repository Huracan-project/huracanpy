from datetime import timedelta
import warnings

import cftime
from dateutil.parser import parse
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime

from . import _csv, track_files, _netcdf, _tempestextremes, witrack, old_hurdat, iris_tc
from . import ibtracs
from .._concat import concat_tracks


rename_defaults = dict(
    longitude="lon",
    latitude="lat",
    # Names for MIT netCDF
    lon_track="lon",
    lat_track="lat",
    # Possible time names
    iso_time="time",
    isotime="time",
)

pandas_valid_time_labels = [
    "year",
    "years",
    "month",
    "months",
    "day",
    "days",
    "hour",
    "hours",
    "minute",
    "minutes",
    "second",
    "seconds",
    "ms",
    "millisecond",
    "milliseconds",
    "us",
    "microsecond",
    "microseconds",
    "ns",
    "nanosecond",
    "nanoseconds",
]


def load(
    filename=None,
    source=None,
    variable_names=None,
    rename=dict(),
    units=None,
    baselon=None,
    infer_track_id=None,
    track_id_prefix=None,
    ibtracs_subset="wmo",
    iris_timestep=timedelta(hours=3),
    tempest_extremes_unstructured=False,
    tempest_extremes_header_str="start",
    track_calendar=None,
    **kwargs,
):
    """Load track data

    The optional parameters for different sources of tracks (currently **IBTrACS**,
    **TRACK** and **TempestExtremes**) are named {source}_{parameter} (in lower case),
    e.g. "ibtracs_subset".

    Parameters
    ----------
    filename : str or list, optional
        The file or files to be loaded. If `source="ibtracs"`, this is not needed as the
        data is either included in huracanpy or downloaded when called. If the filename
        is provided for an online IBTrACS subset, then the raw downloaded data will be
        saved there.
    source : str, optional
        If the file is not a CSV or NetCDF (identified by the file extension) then the
        source needs to be specified to decide how to load the data

        * **track**
        * **track.tilt**
        * **te**, **tempest**, **tempestextremes**, **uz** (For textual format, not csv)
        * **witrack**
        * **ibtracs**
        * **csv**
        * **netcdf**, **nc** (includes support for CHAZ & MIT-Open file provided
          appropriate variable_names)
        * **old_hurdat**, **ecmwf**
        * **iris**

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
        A mapping of variable names to rename (for netCDF files only). Defaults are

        * longitude -> lon
        * latitude -> lat

        To override any of these defaults, you can pass the same name, e.g.

        >>> tracks = huracanpy.load(..., rename=dict(longitude="longitude"))

    units : dict, optional
        A mapping of variable names to units

        >>> tracks = huracanpy.load(..., units=dict(vorticity="s**-1", slp="hPa"))

    baselon : scalar, optional
        Force the loaded longitudes into the range (baselon, baselon + 360). e.g.
        (0, 360) or (-180, 180)

    infer_track_id : list, optional
        If track_id is not a variable in the file, but the individual tracks can be
        inferred from the combination of other variables, e.g. the file has year and
        storm number by year, then pass a list with those variable names, and a new
        track_id variable will be created

    track_id_prefix : str, optional
        If loading from a list of tracks. Add this formattable prefix to the start of
        the track_ids to keep them as unique identifiers. See
        :py:func:`huracanpy.concat_tracks` for more details

    ibtracs_subset : str, default="wmo"
        IBTrACS subset. Two offline versions are available:

        * **wmo**: Data with the wmo_* variables. The data as reported by the WMO agency
          responsible for each basin, so methods are not consistent across basins
        * **usa** or **JTWC**: Data with the usa_* variables. The data as recorded by
          the USA/Joint Typhoon Warning Centre. Methods are consistent across basins,
          but may not be complete.

        To download online data, the subsets are the different files provided by
        IBTrACS.

        * **ACTIVE**: TCs currently active
        * **ALL**: Entire IBTrACS database
        * Specific basins: **EP**, **NA**, **NI**, **SA**, **SI**, **SP**, **WP**
        * **last3years**: self-explanatory
        * **since1980**: Entire IBTrACS database since 1980 (advent of satellite era,
          considered reliable from then on)

    iris_timestep : int or datetime.timedelta, default=datetime.timedelta(hours=3)
        The timestep used in the Imperial College Storm Model (IRIS). This is 3-hours
        in the paper
        (https://www.nature.com/articles/s41597-024-03250-y/tables/1), so unlikely to
        need changing, but provided here in case it does.

    tempest_extremes_unstructured : bool, default=False
        By default the first two columns in TempestExtremes files are the i, j indices
        of the closest gridpoint, but for unstructured grids it is a single lookup index
        so there is one less column
    tempest_extremes_header_str : str, default="start"
        This is an option in the Colin's load function, so I assume this can change
        between files

    track_calendar : [str, tuple] optional
          When loading data from a TRACK ASCII file, if the data uses a different
          calendar to the default :class:`datetime.datetime`, then you can pass this
          argument to load the times in as :class:`cftime.datetime` with the given
          calendar instead

          If the TRACK file has timesteps instead of dates. Then you can pass a tuple
          with the initial time and timestep e.g. ("1940-01-01", 6)

          * The first argument is the initial time and needs to be something readable by
            :class:`numpy.datetime64`, or you can explicity pass a
            :class:`numpy.datetime64` object.
          * The second argument is the step and is passed to :class:`numpy.timedelta64`
            and is assumed to be in hours, or you can explicitly pass a
            :class:`numpy.timedelta64` object and specify the units

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
    # Overwrite default arguments with explicit arguments passed to rename by putting
    # "rename" second in this dictionary combination
    rename = {**rename_defaults, **rename}

    if isinstance(filename, (list, tuple, np.ndarray)):
        # Loop through all the files and open them
        tracks = [
            load(
                filename=f,
                source=source,
                variable_names=variable_names,
                rename=rename,
                units=units,
                baselon=baselon,
                infer_track_id=infer_track_id,
                ibtracs_subset=ibtracs_subset,
                iris_timestep=iris_timestep,
                tempest_extremes_unstructured=tempest_extremes_unstructured,
                tempest_extremes_header_str=tempest_extremes_header_str,
                track_calendar=track_calendar,
                **kwargs,
            )
            for f in filename
        ]

        return concat_tracks(tracks, prefix=track_id_prefix)

    # If source is not given, try to derive the right function from the file extension
    if source is None:
        extension = filename.split(".")[-1]
        if extension == "csv":
            tracks = _csv.load(filename, **kwargs)
        elif extension == "parquet":
            tracks = _csv.load(filename, load_function=pd.read_parquet, **kwargs)
        elif filename.split(".")[-1] == "nc":
            tracks = _netcdf.load(filename, **kwargs)
        else:
            raise ValueError("Source is set to None and file type is not detected")

    # If source is given, use the relevant function
    else:
        source = source.lower()
        if source == "track":
            tracks = track_files.load(filename, variable_names=variable_names)
        elif source == "track.tilt":
            tracks = track_files.load_tilts(filename)
        elif source in ["csv", "uz"]:
            tracks = _csv.load(filename, **kwargs)
        elif source in ["te", "tempest", "tempestextremes"]:
            tracks = _tempestextremes.load(
                filename,
                variable_names,
                tempest_extremes_unstructured,
                tempest_extremes_header_str,
            )
        elif source == "witrack":
            tracks = witrack.load(filename)
        elif source == "ibtracs":
            tracks = ibtracs.load(ibtracs_subset, filename, **kwargs)
        elif source == "netcdf":
            tracks = _netcdf.load(filename, **kwargs)
        elif source in [
            "old_hurdat",
            "ecmwf",
        ]:
            tracks = old_hurdat.load(filename)
        elif source == "iris":
            tracks = iris_tc.load(filename, iris_timestep, **kwargs)
        else:
            raise ValueError(f"Source {source} unsupported or misspelled")

    # xarray.Dataset.rename only accepts keys that are actually in the dataset
    # Also, don't rename to a variable that already exists
    rename = {
        key: rename[key]
        for key in rename
        if key in tracks and rename[key] not in tracks
    }

    if len(rename) > 0:
        tracks = tracks.rename(rename)

    # Time attribute
    tracks = _parse_dates(tracks, calendar=track_calendar)

    # Convert any variables that are objects to explicit string objects.
    # This can cause strings stored as "NA", such as for basin to be converted to NaNs
    # Revert any nans back to their original values
    # Better to do it here than getting caught out later
    for var in tracks:
        if np.issubdtype(tracks[var].dtype, np.object_) and isinstance(
            tracks[var].values[0], str
        ):
            new_var = tracks[var].astype(str)
            false_nans = new_var == "nan"
            new_var[false_nans] = tracks[var][false_nans]
            tracks[var] = new_var

    if units is not None:
        for varname in units:
            tracks[varname].attrs["units"] = units[varname]

    if baselon is not None:
        tracks["lon"] = ((tracks.lon - baselon) % 360) + baselon

    if infer_track_id is not None:
        tracks = tracks.hrcn.add_inferred_track_id(*infer_track_id)

    tracks.track_id.attrs["cf_role"] = "trajectory_id"

    return tracks


def _parse_dates(tracks, calendar):
    if "time" in tracks:
        time = tracks.time
        # If the time has already been correctly parsed just return the tracks as is
        if np.issubdtype(time.dtype, np.datetime64):
            return tracks

        # Track YYYYMMDDHH format not interpreted automatically. Change to str with
        # isoformat
        if (
            np.issubdtype(time.dtype, np.integer)
            and (time.astype(str).str.len() == 10).all()
            or np.issubdtype(time.dtype, np.str_)
            and (time.str.len() == 10).all()
        ):
            time = tracks.time.astype(str)
            time = time.str.slice(0, 4).str.cat(
                "-",
                time.str.slice(4, 6),
                "-",
                time.str.slice(6, 8),
                " ",
                time.str.slice(8, 10),
                ":00",
            )

        if isinstance(calendar, (tuple, list)):
            # Time is integer timesteps
            initial_date = np.datetime64(calendar[0])
            timestep = calendar[1]
            if not isinstance(timestep, np.timedelta64):
                timestep = np.timedelta64(timestep, "h")
            return tracks.assign(time=initial_date + (time - 1) * timestep)

        elif isinstance(calendar, str):
            # cftime calendar
            default = cftime.datetime(1, 1, 1, calendar=calendar)
            time = [parse(t, default=default) for t in time.values]
            return tracks.assign(time=("record", time))

        # Convert strings to np.datetime64, but allow for varying precision for possible
        # out of bounds times
        elif isinstance(time.values[0], str):
            # This may still break at this point with older versions of xarray
            # attempting to convert back to "ns" precision
            try:
                newtime = time.astype("datetime64")

                if (time.astype("datetime64[Y]").dt.year != newtime.dt.year).any():
                    raise OutOfBoundsDatetime

                return tracks.assign(time=newtime)
            except OutOfBoundsDatetime:
                warnings.warn(
                    "Converting out of bounds np.datetime64 to cftime.datetime. Update"
                    " to xarray>=2025.01.2 to remove this warning and use lower"
                    " precision np.datetime64 instead"
                )
                time = [parse(t) for t in time.values]
                time = [
                    cftime.datetime(
                        t.year,
                        t.month,
                        t.day,
                        t.hour,
                        t.minute,
                        t.second,
                        t.microsecond,
                    )
                    for t in time
                ]
                return tracks.assign(time=("record", time))

        else:
            # Not interpretable as datetime, just return as is
            return tracks

    else:
        # Combine separate year/month/day etc. values into a time, and drop those
        # variables from the Dataset
        time_vars = {
            var: tracks[var].values for var in tracks if var in pandas_valid_time_labels
        }
        return tracks.assign(time=("record", pd.to_datetime(time_vars))).drop_vars(
            list(time_vars.keys())
        )
