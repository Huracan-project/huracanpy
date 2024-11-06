.. huracanpy documentation master file, created by
   sphinx-quickstart on Fri May 17 13:33:16 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HuracanPy
=========

A python package for working with various forms of feature tracking data, including but not restricted to cyclone tracks.

Why HuracanPy?
--------------
The idea of this package is to provide a common tool for working with cyclone track data. 
In particular, HuracanPy can read tracks from many different sources/trackers. 
It also provides useful functions to analyse these tracks, including many common diagnostics.
Our goal is to make track data analysis more accessible, and to promote good reproducibility practices.


Installation
------------
To install the package, you can simply use ``pip``::

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
* Load cyclone tracks (`load`);
* Add information to your tracks (`info`);
* Subset (`subset`) and interpolate (`interp`) the tracks;
* Compute standard diagnostic metrics  (`calc`);
* Make simple plots (`plot`);
* Compare several datasets between them (`assess`).
The flowchart below illustrates this structure. 

.. image:: images/package_structure_flowchart/flowchart.png
  :width: 400
  :alt: Alternative text

What you need to know to use huracanpy
--------------------------------------
This package is distributed under the MIT licence. As such, you can [...]

A JOSS paper is currently being prepared that you will be able to cite in your publication where HuracanPy was used.

Please watch the GitHub repository to get information about HuracanPy. 

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
