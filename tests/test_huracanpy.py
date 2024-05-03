import numpy as np

import huracanpy
from huracanpy.utils.geography import get_hemisphere, get_basin
from huracanpy.utils.category import get_sshs_cat, get_pressure_cat

def test_load_track():
    data = huracanpy.load(huracanpy.example_TRACK_file, tracker="TRACK")
    assert len(data.groupby("track_id")) == 2

def test_load_csv():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "tempestextremes")
    assert len(data) == 13
    assert len(data.obs) == 99
    assert len(data.groupby("track_id")) == 3
    
def test_hemisphere():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    assert np.unique(get_hemisphere(data)) == np.array(["S"])
    
def test_basin():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    assert get_basin(data)[0] == "AUS"
    assert get_basin(data)[-1] == "SI"
    
def test_sshs():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    assert get_sshs_cat(data.wind10).min() == -1
    assert get_sshs_cat(data.wind10).max() == 0
    
def test_pressure_cat():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    Klotz = get_pressure_cat(data.slp/100)
    Simps = get_pressure_cat(data.slp/100, convention = "Simpson")
    assert Klotz.sum() == 62
    assert Simps.sum() == -23