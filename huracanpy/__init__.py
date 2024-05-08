"""huracanpy - A python package for working with various forms of feature tracking data"""

__version__ = "0.1.0"
__author__ = "Leo Saffin <l.saffin@reading.ac.uk>, Stella Bourdin <stella.bourdin@physics.ox.ac.uk>, Kelvin Ng "
__all__ = ["load", "save", "utils", "diags"]

import pathlib

from ._tracker_specific import TRACK, csv, netcdf
from . import utils
from . import diags


here = pathlib.Path(__file__).parent
testdata_dir = here / "example_data"

example_TRACK_file = str(
    testdata_dir / "tr_trs_pos.2day_addT63vor_addmslp_add925wind_add10mwind.tcident.new"
)

example_csv_file = str(testdata_dir / "sample.csv")

example_year_file = str(
    testdata_dir / "ERA5_1996_UZ.csv"
)

example_TRACK_netcdf_file = str(
    testdata_dir
    / "tr_trs_pos.2day_addT63vor_addmslp_add925wind_add10mwind.tcident.new.nc"
)


def load(filename, tracker=None, add_info = False, **kwargs):
    # If tracker is not given, try to derive the right function from the file extension
    if (tracker is None):
        if filename.split(".")[-1] == "csv":
            data = csv.load(filename)
        elif filename.split(".")[-1] == "nc":
            data = netcdf.load(filename, **kwargs)
        else :
            raise ValueError(f"{tracker} is set to None and file type is not detected")

    # If tracker is given, use the relevant function
    else:
        if tracker.lower() == "track":

            data = TRACK.load(filename, **kwargs)
        elif tracker.lower() in ["csv", "te", "tempestextremes", "uz"]  :
            data = csv.load(filename)
        else:
            raise ValueError(f"Tracker {tracker} unsupported or misspelled")
            
    if add_info :  # TODO : Manage potentially different variable names
        data["hemisphere"] = utils.geography.get_hemisphere(data.lat)
        data["basin"] = utils.geographyget_basin(data.lon, data.lat)
        data["season"] = utils.time.get_season(data.track_id, data.lat, data.time) 
        if "wind10" in list(data.keys()) : # If 'wind10' is in the attributes
            data["sshs"] = utils.category.get_sshs_cat(data.wind10)
        if "slp" in list(data.keys()) : # If 'slp' is in the attributes
            data["pres_cat"] = utils.category.get_pressure_cat(data.slp)
        
    return data


def save(dataset, filename):
    if filename.split(".")[-1] == "nc":
        netcdf.save(dataset, filename)
    elif filename.split(".")[-1] == "csv":
        dataset.to_dataframe().to_csv(filename, index= False)
    else:
        raise NotImplementedError("File format not recognized. Please use one of {.nc, .csv}")
