"""Module for functions related to the ibtracs database"""

from urllib.request import urlretrieve
import os
import warnings
import pathlib
from contextlib import contextmanager

here = pathlib.Path(__file__).parent
ibdata_dir = here / "_ibtracs_files/"

wmo_file = str(ibdata_dir / "wmo.csv")
usa_file = str(ibdata_dir / "usa.csv")


@contextmanager
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

    yield filename

    if clean:
        os.remove(filename)  # Somehow, this is slower than the rest ofthe function (??)


def offline(subset="wmo"):
    """
    Function to load offline IBTrACS datasets. These are embedded within the package, rather than downloaded online.
    Because of the offline nature of this feature, the datasets might be outdated depending on your last update of the
    package and/or the last update of the datasets by the developers.
    Last update by the developers for this version dates back to May 24th, 2024.

    In order to reduce the size of the embedded files, filtering was made both on the attributes and on the records.
    All offline datasets are based on the "since1980" subset of IBTrACS.
    Recent seasons with tracks marked as "provisionnal" were removed. All spur tracks were removed.
    Only 6-hourly data is provided.

    Two subsets are currently available:
        * "wmo" contains the data provided in the "wmo" columns, which correspond to the data provided by the center
          responsible for the area of a given point. (see https://community.wmo.int/en/tropical-cyclone-regional-bodies)
          Note that within this dataset, wind units are not homogeneous: they are provided as collected from the
          meteorological agencies, which means that they have different time-averaging for wind extrema.
        * "usa" contains the data provided in the "wmo" columns, which is provided by the NHC or the JTWC.

    For more information, you may see the IBTrACS column documentation at
    https://www.ncei.noaa.gov/sites/default/files/2021-07/IBTrACS_v04_column_documentation.pdf

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
        return wmo_file
    if subset.lower() in ["usa", "jtwc"]:
        return usa_file


# TODOS:
# Make warnings better
# Deal with units, in general
