"""
Module to load tracks stored as csv files, including TempestExtremes output.
"""

import pandas as pd

# All values recognised as NaN by pandas.read_csv, except "NA" which we want to load
# normally because it is a basin, and added "" to interpret empty entries as NaN
pandas_na_values = [
    " ",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null ",
    "",
]

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
    filename,
    load_function=pd.read_csv,
    **kwargs,
):
    """Load csv tracks data as an xarray.Dataset
    These tracks may come from TempestExtremes StitchNodes, or any other source.

    Parameters
    ----------
    filename : str
        The file must contain at least longitude, latitude, time and track ID.
            - longitude and latitude can be named that, or lon and lat.
            - time must be defined a single `time`column or by four columns : year, month, day, hour
            - track ID must be within a column named track_id.

    load_function : callable
        One of the load functions in pandas

    **kwargs
        Remaining keywords are passed to the pandas

    Returns
    -------
    xarray.Dataset
    """
    # Update keywords with extra defaults for dealing with "NA" as basin not nan
    # Put kwargs second in this statement, so it can override defaults
    if load_function is pd.read_csv:
        kwargs = {**dict(na_values=pandas_na_values, keep_default_na=False), **kwargs}

    ## Read file
    tracks = load_function(filename, **kwargs)
    # Remove leading/trailing spaces and make all column names lowercase
    tracks.columns = tracks.columns.str.strip().str.lower()

    ## Time attribute
    if "iso_time" in tracks.columns:
        tracks["time"] = pd.to_datetime(tracks.iso_time)
        tracks = tracks.drop(columns="iso_time")
    elif "time" in tracks.columns:
        tracks["time"] = pd.to_datetime(tracks.time)
    else:
        # Combine separate year/month/day etc. values into a time, and drop those
        # variables from the dataframe
        time_vars = [var for var in tracks.columns if var in pandas_valid_time_labels]
        tracks["time"] = pd.to_datetime(tracks[time_vars])
        tracks = tracks.drop(time_vars, axis="columns")

    # Output xr dataset
    tracks = tracks.to_xarray().rename({"index": "record"}).drop_vars("record")

    # Convert any variables that are objects to explicit string objects.
    # This can cause strings stored as "NA", such as for basin to be converted to NaNs
    # Revert any nans back to their original values
    # Better to do it here than getting caught out later
    for var in tracks:
        if tracks[var].dtype == "O" and isinstance(tracks[var].data[0], str):
            new_var = tracks[var].astype(str)
            false_nans = new_var == "nan"
            new_var[false_nans] = tracks[var][false_nans]
            tracks[var] = new_var

    return tracks
