==========
User Guide
==========

Pre-requisites
--------------

xarray
~~~~~~

Our package is designed to load track data as xarray objects. 
We recommend you get familiar with xarray to make the most of this package (https://tutorial.xarray.dev/intro.html). 


Package structure
-----------------

The package has several module that allow you to:

* Load cyclone tracks (`load`);
* Subset (`subset`) and interpolate (`interp`) the tracks;
* Add information to your tracks (`info`);
* Compute standard diagnostic metrics (`calc`);
* Make simple plots (`plot`);
* Compare several datasets between them (`assess`).

It also have a TC-specific module (`tc`) which contains functions for ACE (Accumulated Cyclonic Energy) and specific categories used for tropical cyclones.

The flowchart below illustrates this structure. The present user guide details how to use each of these modules.

.. image:: ../images/package_structure_flowchart/flowchart.png
  :width: 1000
  :alt: Alternative text

xarray accessor
---------------

The package also defines a `.hrcn` xarray accessor, which allow you to call most of the functions as methods on the dataset itself. 


.. toctree::
    :maxdepth: 4
    :hidden:

    self
    install
    demo
    load
    save
    info
    subset
    interp
    calc
    tc
    plot
    assess
    speed
    accessor