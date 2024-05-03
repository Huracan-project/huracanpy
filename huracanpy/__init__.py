"""huracanpy - A python package for working with various forms of feature tracking data"""

__version__ = "0.1.0"
__author__ = "Leo Saffin <l.saffin@reading.ac.uk>, Stella Bourdin <stella.bourdin@physics.ox.ac.uk>, Kelvin Ng "
__all__ = ["load"]

import pathlib

from ._tracker_specific import TRACK, csv


here = pathlib.Path(__file__).parent
testdata_dir = here / "example_data"

example_TRACK_file = str(
    testdata_dir / "tr_trs_pos.2day_addT63vor_addmslp_add925wind_add10mwind.tcident.new"
)

example_csv_file = str(
    testdata_dir / "sample.csv"
)


def load(filename, tracker=None, **kwargs):
    if tracker.lower() == "track":
        return TRACK.load(filename, **kwargs)
    if (tracker.lower() in ["csv", "te", "tempestextremes", "uz"]) or ((tracker == None) &  (filename[-3:] == "csv")):
        return csv.load(filename)
    else:
        raise ValueError(f"Tracker {tracker} unsupported or misspelled")

