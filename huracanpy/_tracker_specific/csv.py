"""
Module to load tracks stored as csv files, including TempestExtremes output.
"""

import pandas as pd


from .. import utils


def load(
    filename,
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

    Returns
    -------
    xarray.Dataset
    """

    ## Read file
    tracks = pd.read_csv(filename)
    if (
        tracks.columns.str[0][1] == " "
    ):  # Sometimes columns names are read starting with a space, which we remove
        tracks = tracks.rename(columns={c: c[1:] for c in tracks.columns[1:]})
    tracks = tracks.rename(
        {"longitude": "lon", "latitude": "lat"}
    )  # Rename lon & lat columns if necessary

    ## Geographical attributes
    if "lon" in tracks.columns:
        tracks.loc[tracks.lon < 0, "lon"] += (
            360  # Longitude are converted to [0,360] if necessary
        )

    ## Time attribute
    if "time" not in tracks.columns:
        tracks["time"] = utils.time.get_time(
            tracks.year, tracks.month, tracks.day, tracks.hour
        )
    else:
        tracks["time"] = pd.to_datetime(tracks.time)

    # Output xr dataset
    return tracks.to_xarray().rename({"index": "obs"})
