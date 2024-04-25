"""huracanpy - A python package for working with various forms of feature tracking data"""

__version__ = "0.1.0"
__author__ = "Leo Saffin <l.saffin@reading.ac.uk>, Stella Bourdin, Kelvin Ng "
__all__ = ["load"]

import pathlib

from ._tracker_specific import TRACK


here = pathlib.Path(__file__).parent
testdata_dir = here / "example_data"

example_TRACK_file = str(
    testdata_dir / "tr_trs_pos.2day_addT63vor_addmslp_add925wind_add10mwind.tcident.new"
)


def load(filename, tracker=None, **kwargs):
    if tracker.lower() == "track":
        return TRACK.load(filename, **kwargs)
    else:
        raise ValueError(f"Tracker {tracker} unsupported or misspelled")
