from . import _csv
from . import _TRACK
from . import _netcdf
from . import ibtracs
from huracanpy import utils


def load(
    filename=None,
    tracker=None,
    add_info=False,
    ibtracs_online=False,
    ibtracs_subset="ALL",
    ibtracs_clean=True,
    **kwargs,
):
    """

    Parameters
    ----------
    filename : str
    tracker : str
    add_info : bool, default=False
    ibtracs_online : bool, default=False
        * False: Use a small subset of the IBTrACS data included in this package
        * True: Download the IBTrACS data
    ibtracs_subset : str, default="ALL"
        IBTrACS subset. Can be one of
        * ACTIVE: TCs currently active
        * ALL: Entire IBTrACS database
        * Specific basins: EP, NA, NI, SA, SI, SP, WP
        * last3years: self-explanatory
        * since1980: Entire IBTrACS database since 1980 (advent of satellite era, considered reliable from then on)
    ibtracs_clean : bool, default=True
        If downloading IBTrACS data, this parameter says whether to delete the
        downloaded file after loading it into memory.
    **kwargs

    Returns
    -------
    xarray.Dataset

    """
    # If tracker is not given, try to derive the right function from the file extension
    if tracker is None:
        if filename.split(".")[-1] == "csv":
            data = _csv.load(filename)
        elif filename.split(".")[-1] == "nc":
            data = _netcdf.load(filename, **kwargs)
        else:
            raise ValueError(f"{tracker} is set to None and file type is not detected")

    # If tracker is given, use the relevant function
    else:
        if tracker.lower() == "track":
            data = _TRACK.load(filename, **kwargs)
        elif tracker.lower() in ["csv", "te", "tempestextremes", "uz"]:
            data = _csv.load(filename)
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
