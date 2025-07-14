import re
from urllib.request import urlretrieve

import pandas as pd

from . import _csv

path = "https://raw.githubusercontent.com/tenkiman/superBT-V04/refs/heads/v04/dat/"
# Currently headers are broken for sbt file
# meta_fname = "h-meta-sbt-v04-vars.csv"
# tracks_fname = "sbt-v04-2007-2022-MRG.csv"

# Use the all file for now
# TODO add options for different files/time periods
meta_fname = "h-meta-md3-vars.csv"
tracks_fname = "all-md3-2007-2022-MRG.csv"
units_rename = dict(degN="degrees_north", degE="degrees_east")


def load():
    filename, _ = urlretrieve(path + meta_fname, None)  # noqa: S310
    # Use custom separator because some variable details also have a comma within the
    # quotes. Second line removes the quotes at the other end
    meta = pd.read_csv(filename, names=["varname", "details"]).replace(
        "osname", "sname"
    )
    meta.details = meta.details.apply(lambda x: x.split("'")[1])

    filename, _ = urlretrieve(path + tracks_fname, None)  # noqa: S310
    tracks = _csv.load(filename).drop_vars("unnamed: 33")

    for n, row in meta.iterrows():
        varname = row.varname
        details = row.details

        # Check if variable description has units in square brackets
        m = re.match(r"(.*) \[(\w+)\]", details)
        if m and len(m.groups()) == 2:
            details, units = m.groups()
            if units in units_rename:
                units = units_rename[units]
            tracks[varname].attrs["units"] = units

        tracks[varname].attrs["description"] = details

    return tracks
