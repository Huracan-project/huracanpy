"""Module for functions related to the ibtracs database"""

from urllib.request import urlretrieve
import os
import numpy as np
import warnings
import pathlib
from .. import load
from .. import save

here = pathlib.Path(__file__).parent
ibdata_dir = here / "_ibtracs_files/"

wmo_file = str(ibdata_dir / "wmo.csv")
usa_file = str(ibdata_dir / "usa.csv")


def online(subset, filename="ibtracs.csv", clean=True):
    """
    Downloads an load into the current workspace the specified ibtracs subset from the IBTrACS archive online.

    Parameters
    ----------
    subset : str
    IBTrACS subset. Can be one of
    * ACTIVE: TCs currently active
    * ALL: Entire IBTrACS database
    * Specific basins: EP, NA, NI, SA, SI, SP, WP
    * last3years: self-explanatory
    * since1980: Entire IBTrACS database since 1980 (advent of satellite era, considered reliable from then on)

    filename : str
    (temporary) file to which to save the data
    The default is "tmp/ibtracs_ACTIVE.csv".

    clean : bool
    If True (default), remove the temporary file after loading the data.

    Returns
    -------
    ib : the IBTrACS subset requested

    """
    # TODO: Make it so that the user does not need to specify the filename
    url = (
        "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs."
        + subset
        + ".list.v04r00.csv"
    )
    urlretrieve(url, filename)
    ib = load(filename, tracker="ibtracs")
    if clean:
        os.remove(filename)  # Somehow, this is slower than the rest ofthe function (??)
    return ib


def _prepare_offline(wmo=True, usa=True):
    ib = online("since1980", "tmp/ibtracs.csv")

    # Remove season with tracks that are still provisional
    first_season_provi = ib.where(
        ib.track_type == "PROVISIONAL", drop=True
    ).season.min()
    ib = ib.where(ib.season < first_season_provi, drop=True)

    # Remove spur tracks
    ib = ib.where(ib.track_type == "main", drop=True)  # 348MB

    # - WMO subset
    if wmo:
        print("... WMO ...")
        ## Select WMO variables
        ib_wmo = ib[
            ["sid", "season", "basin", "time", "lon", "lat", "wmo_wind", "wmo_pres"]
        ].rename({"sid": "track_id", "wmo_wind": "wind", "wmo_pres": "slp"})  # 19MB

        ## Select only 6-hourly time steps
        ib_wmo = ib_wmo.where(ib_wmo.time.dt.hour % 6 == 0, drop=True)  # 9MB

        ## Deal with var types to reduce size ( at the moment, reduces by 42% )
        for var in ["lat", "lon", "slp", "wind"]:
            ib_wmo[var] = ib_wmo[var].astype(np.float16)
        for var in [
            "season",
        ]:
            ib_wmo[var] = ib_wmo[var].astype(np.int16)

        ## Save WMO file
        save(ib_wmo, wmo_file)

    if usa:
        # - USA subset
        print("... USA ...")
        ## Select USA variables
        ib_usa = ib[
            [
                "sid",
                "season",
                "basin",
                "time",
                "usa_lat",
                "usa_lon",
                "usa_status",
                "usa_wind",
                "usa_pres",
                "usa_sshs",
            ]
        ].rename(
            {
                "sid": "track_id",
                "usa_lat": "lat",
                "usa_lon": "lon",
                "usa_status": "status",
                "usa_wind": "wind",
                "usa_pres": "slp",
                "usa_sshs": "sshs_cat",
            }
        )  # 23MB

        ## Select only 6-hourly time steps
        ib_usa = ib_usa.where(ib_usa.time.dt.hour % 6 == 0, drop=True)  # 11MB

        ## Remove lines with no data
        ib_usa = ib_usa.where(~np.isnan(ib_usa.lon), drop=True)

        ## Deal with var types to reduce size ( at the moment, reduces by 25% ) -> TODO : Manage wind and slp data...
        for var in ["lat", "lon", "wind", "slp"]:
            ib_usa[var] = ib_usa[var].astype(np.float16)
        for var in ["season"]:
            ib_usa[var] = ib_usa[var].astype(np.int16)
        for var in ["sshs_cat"]:
            ib_usa[var] = ib_usa[var].astype(np.int8)

        ## Save
        save(ib_usa, usa_file)

    warnings.warn(
        "If you just updated the offline files within the package, do not forget to update information in offline loader warnings"
    )


def offline(subset="wmo"):
    """
    Function to load offline IBTrACS datasets. These are embedded within the package, rather than downloaded online.
    Because of the offline nature of this feature, the datasets might be outdated depending on your last update of the package and/or the last update of the datasets by the developers.
    Last update by the developers for this version dates back to May 24th, 2024.

    In order to reduce the size of the embedded files, filtering was made both on the attributes and on the records.
    All offline datasets are based on the "since1980" subset of IBTrACS.
    Recent seasons with tracks marked as "provisionnal" were removed. All spur tracks were removed.
    Only 6-hourly data is provided.

    Two subsets are currently available:
        * "wmo" contains the data provided in the "wmo" columns, which correspond to the data provided by the center responsible for the area of a given point. (see https://community.wmo.int/en/tropical-cyclone-regional-bodies)
                Note that within this dataset, wind units are not homogeneous: they are provided as collected from the meteorological agencies, which means that they have different time-averaging for wind extrema.
        * "usa" contains the data provided in the "wmo" columns, which is provided by the NHC or the JTWC.

    For more information, you may see the IBTrACS column documentation at https://www.ncei.noaa.gov/sites/default/files/2021-07/IBTrACS_v04_column_documentation.pdf

    Parameters
    ----------
    subset : str, optional
        "usa" or "wmo". The default is "wmo".

    Returns
    -------
    xr.Dataset
        The set of tracks requested.

    """
    warnings.warn(
        "This offline function loads a light version of IBTrACS which is embedded within the package, based on a file produced manually by the developers.\n\
                  It was last updated on the 24nd May 2024, based on the IBTrACS file at that date.\n\
                  It contains only data from 1980 up to the last year with no provisional tracks. All spur tracks were removed. Only 6-hourly time steps were kept."
    )
    if subset.lower() == "wmo":
        warnings.warn(
            "You are loading the IBTrACS-WMO subset. \
                      This dataset contains the positions and intensity reported by the WMO agency responsible for each basin\n\
                      Be aware of the fact that wind and pressure data is provided as they are in IBTrACS, \
                      which means in particular that wind speeds are in knots and averaged over different time periods.\n\
                    For more information, see the IBTrACS column documentation at https://www.ncei.noaa.gov/sites/default/files/2021-07/IBTrACS_v04_column_documentation.pdf"
        )
        return load(wmo_file)
    if subset.lower() in ["usa", "jtwc"]:
        return load(usa_file)


# TODOS:
# Make warnings better
# Deal with units, in general
