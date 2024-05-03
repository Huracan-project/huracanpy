import numpy as np

import huracanpy

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
    assert np.unique(huracanpy.utils.geography.get_hemisphere(data)) == np.array(["S"])
    
def test_basin():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    assert huracanpy.utils.geography.get_basin(data)[0] == "AUS"
    assert huracanpy.utils.geography.get_basin(data)[-1] == "SI"
    
def test_sshs():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    assert huracanpy.utils.category.get_sshs_cat(data.wind10).min() == -1
    assert huracanpy.utils.category.get_sshs_cat(data.wind10).max() == 0
    
def test_pressure_cat():
    data = huracanpy.load(huracanpy.example_csv_file, tracker = "csv")
    Klotz = huracanpy.utils.category.get_pressure_cat(data.slp/100)
    Simps = huracanpy.utils.category.get_pressure_cat(data.slp/100, convention = "Simpson")
    assert Klotz.sum() == 62
    assert Simps.sum() == -23