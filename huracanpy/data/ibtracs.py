"""Module for functions related to the ibtracs database"""

from urllib.request import urlretrieve
import os
import numpy as np
import warnings
import pathlib
from .. import load
from .. import save

here = pathlib.Path(__file__).parent
data_dir = here / "_ibtracs_files/"


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


def _prepare_offline():
    ib = online("since1980", "tmp/ib3years.csv")

    # Remove season with tracks that are still provisional
    first_season_provi = ib.where(
        ib.track_type == "PROVISIONAL", drop=True
    ).season.min()
    ib = ib.where(ib.season < first_season_provi, drop=True)

    # Remove spur tracks
    ib = ib.where(ib.track_type == "main", drop=True)

    # - WMO subset
    print("... WMO ...")
    ## Select WMO variables
    ib_wmo = ib[
        ["sid", "season", "basin", "name", "time", "lon", "lat", "wmo_wind", "wmo_pres"]
    ]

    # Deal with missing values
    # ib_wmo = ib_wmo.where(ib_wmo != ' ')

    ## Deal with var types to reduce size ( at the moment, reduces by 25% ) -> TODO : Manage wind and slp data...
    for var in ["lat", "lon"]:
        ib_wmo[var] = ib[var].astype(np.float16)
    for var in ["season"]:
        ib_wmo[var] = ib[var].astype(np.int16)

    ## Save WMO file
    save(ib_wmo, "huracanpy/data/_ibtracs_files/wmo.csv")

    # - USA subset
    print("... USA ...")
    ## Select USA variables
    # C = np.array(list(ib.keys())) # Variable names
    # C = C[[s.startswith("usa") for s in C]] # Variable names starting with usa
    # ib_usa = ib[["sid", "season", "basin", "name",] + list(C)]
    ib_usa = ib[
        [
            "sid",
            "season",
            "basin",
            "name",
            "time",
            "usa_lat",
            "usa_lon",
            "usa_status",
            "usa_wind",
            "usa_pres",
            "usa_sshs",
        ]
    ]

    ## Deal with var types to reduce size ( at the moment, reduces by 25% ) -> TODO : Manage wind and slp data...
    for var in ["lat", "lon"]:
        ib_usa[var] = ib[var].astype(np.float16)
    for var in ["season"]:
        ib_usa[var] = ib[var].astype(np.int16)

    ## Save
    save(ib_usa, "huracanpy/data/_ibtracs_files/usa.csv")

    warnings.warn(
        "If you just updated the offline files within the package, do not forget to update information in offline loader warnings"
    )


def offline(subset="wmo"):
    warnings.warn(
        "This offline function loads a light version of IBTrACS which is embedded within the package, based on a file produced manually by the developers.\n\
                  It was last updated on the 21st May 2024, based on the IBTrACS file at that date.\n\
                  It contains only data from 1980 up to the last year with no provisional tracks. All spur tracks were removed."
    )
    if subset.lower() == "wmo":
        warnings.warn(
            "You are loading the IBTrACS-WMO subset. \
                      This dataset contains the positions and intensity reported by the WMO agency responsible for each basin\n\
                      Be aware of the fact that wind and pressure data is provided as they are in IBTrACS, \
                      which means in particular that wind speeds are in knots and averaged over different time periods.\n\
                    For more information, see the IBTrACS column documentation at https://www.ncei.noaa.gov/sites/default/files/2021-07/IBTrACS_v04_column_documentation.pdf"
        )
        return load(str(data_dir / "wmo.csv"))
    if subset.lower() in ["usa", "jtwc"]:
        return load(str(data_dir / "usa.csv"))


# TODOS:
# Deal with NA being read as NaN
# Make smaller files
# Make warnings better
# Deal with units, in general
# Remove lines with no data from WMO and USA
