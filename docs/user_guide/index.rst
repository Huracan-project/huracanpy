User Guide
==========

Installation
------------
To install the package, you can use ``pip``::

    pip install huracanpy

This can fail with older python versions due to issues with installing cartopy through
pip. If this happens, use conda to install cartopy first
(e.g. ``conda install -c conda-forge cartopy``), then install huracanpy as normal

Usage
-----
The idea of this package is to be a standard way for working with cyclone track data. We
were all working on track data, but in slightly different ways which makes sharing code
more difficult. The method chosen here is to treat a set of tracks as an `xarray.Dataset <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`_ with a
minimal number of assumed variables (track_id, lon, lat, time). e.g. running

.. code-block:: python

   import huracanpy

   tracks = huracanpy.load(huracanpy.example_csv_file)
   minimal_tracks = tracks[["track_id", "lon", "lat", "time"]]

   print(minimal_tracks)

gives

.. code-block::

   <xarray.Dataset>
   Dimensions:   (record: 99)
   Coordinates:
     * record    (record) int64 0 1 2 3 4 5 6 7 8 9 ... 89 90 91 92 93 94 95 96 97 98
   Data variables:
       track_id  (record) int64 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 2 2 2 2 2 2 2 2 2 2 2 2
       lon       (record) float64 120.5 119.0 119.0 119.2 ... 57.5 58.5 60.25 62.25
       lat       (record) float64 -14.25 -14.75 -15.0 -15.0 ... -39.25 -42.0 -45.5
       time      (record) datetime64[ns] 1980-01-06T06:00:00 ... 1980-01-30T18:00:00

where each "record" corresponds to a TC point (time, lon, lat).
All variables that were present in your file are variables in the loaded dataset.

Note that the data is one dimensional but represents multiple tracks. This is done
rather than having track_id as an additional dimension to avoid having to add a bunch of
extra blank data to each track when they are not the same length. The ``groupby`` function
allows us to easily loop over or index tracks in this format.

.. code-block:: python

   import huracanpy

   tracks = huracanpy.load(huracanpy.example_csv_file)

   track_groups = tracks.groupby("track_id")

   # Selecting track by ID
   # The track_id is not necessarily an integer, it follows whatever you have loaded
   # e.g. could be a string for IBTrACS
   track_id1 = track_groups[1]

   # Iterating over all tracks
   # Each track will be a subset of the xarray Dataset with a unique track_id
   for track_id, track in track_groups:
       # Do something with the track


csv tracks data
^^^^^^^^^^^^^^^
If you tracks are stored in csv (including if they were outputed from TempestExtremes' StitchNodes),
you can specify the ``tracker="csv"`` argument, or, if your filename ends with *csv*, it will be detected automatically.

TRACK tracks data
^^^^^^^^^^^^^^^^^
If your tracks are in TRACK format, use the `tracker="TRACK"` option

.. toctree::
    :maxdepth: 4
    :hidden:

    self
    extract
