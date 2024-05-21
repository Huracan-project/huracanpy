"""Module for functions related to the ibtracs database"""

from urllib.request import urlretrieve
import os
from .. import load


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
