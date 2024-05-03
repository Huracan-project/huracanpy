# huracanpy

A python package for working with various forms of feature tracking data

## Installation
To install the package, you can use `pip`: `pip install huracanpy`

## Usage
### Load tracks data 

Use the generic `load` function:
```
import huracanpy
tracks = huracanpy.load("data.csv")
```

Your `tracks` object will be an uni-dimensional pandas dataset, where each "obs" corresponds to a TC point (time, lon, lat). 
All attributes taht were present in your file are variable in that tracks dataset.

#### csv tracks data
If you tracks are stored in csv (including if they were outputed from TempestExtremes' StitchNodes), 
you can specify the `tracker="csv"` argument, or, if your filename ends with `csv`, it will be detected automatically.

#### TRACK tracks data
If your tracks are in TRACK format, use the `tracker="TRACK"` option

