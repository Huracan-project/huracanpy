.. huracanpy documentation master file, created by
   sphinx-quickstart on Fri May 17 13:33:16 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HuracanPy
=========

A python package for working with various forms of feature tracking data, including but not restricted to cyclone tracks.

Statement of need
-----------------
The idea of this package is to be a standard way for working with cyclone track data. We
were all working on track data, but in slightly different ways which makes sharing code
more difficult.

Installation
------------
To install the package, you can use ``pip``::

    pip install huracanpy

This can fail with older python versions due to issues with installing cartopy through
pip. If this happens, use conda to install cartopy first
(e.g. ``conda install -c conda-forge cartopy``), then install huracanpy as normal

About xarray
------------
Our package is designed to load track data as xarray objects. 
We recommend you get familiar with xarray to make the most of this package (https://tutorial.xarray.dev/intro.html). 

Package structure
-----------------

The package has several module that allow you to:
* Add information to your tracks (e.g. geographical info, translation speed, etc.);
* Subset part of the tracks;
* Compute standard diagnostic metrics;
* Make simple plots;
* Compare several datasets between them.
The flowchart below illustrates this structure. 

.. image:: path/filename.png
  :width: 400
  :alt: Alternative text

What you need to know to use huracanpy
--------------------------------------
This package is distributed under the MIT licence. As such, you can [...]

A JOSS paper is currently being prepared that you will be able to cite in your publication where HuracanPy was used.

Please subscribe to the mailing list [TBA] to get information about HuracanPy. 

If you encounter any problem while using HuracanPy, please create an issue in GitHub (or consider contributing to the package!) 

Contributing to HuracanPy
-------------------------
Although this package was created within the scope of the Huracán project, anyone is welcome to contribute. 
In particular, we welcome any contributions to extend the package beyond cyclone communities.
Please get in touch if you wish to contribute!



.. toctree::
    :hidden:

    user_guide/index
    examples/index
    api/index
