---
title: 'HuracanPy: A Python package for reading and analysing cyclone tracks'
tags:
  - Python
  - cyclones
authors:
  - name: Stella Bourdin
    orcid: 0000-0003-2635-5654
    equal-contrib: true
    affiliation: 1 
    corresponding: true
  - name: Leo Saffin
    orcid: 0000-0002-6744-7785
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
    corresponding: true
affiliations:
 - name: University of Oxford, Department of Physics, Oxford, UK
   index: 1
   ror: 05v62cm79
 - name: University of Reading, Department of Meteorology, Reading, UK # Please adjust
   index: 2
   ror: 05v62cm79
date: 8 January 2024
bibliography: paper.bib

---

# Summary

HuracanPy is a Python module for working with cyclone track data, where cyclone tracks are recorded as a list of points in space and time. 
HuracanPy provides a powerful loading function that supports all the track data formats the authors are aware of, and loads the tracks as Xarray datasets [@xarray2017].
As such, we draw from the power of Xarray's built-in functions and the functionality of other libraries that work with Xarray, such as MetPy [@metpy]. 
We also provide functions specific to cyclone analysis, which are gathered as methods in an Xarray accessor. 
These include functions to subset or interpolate track data, to add useful information, to compute common diagnoses, and to make quick exploration plots. 
With this package, we hope to make track data analysis more accessible and more reproducible. 

# Defining track data

Cyclones can be defined as pointwise features in space and time. 
Therefore, their trajectory can be recorded as a list of points in space and time. 
We call "track data" a list of such points. 
In many cases, the list not only consist of position in space and time, but also contains additional information about the cyclone, such as intensity. 
When the dataset contains data about several cyclones, each individual track is identified by a "track ID". 

# Statement of need

In meteorological and climatological research about cyclones of any kind, 
we have been using different file formats and conventions for track data. 
HuracanPy's first purpose is to be able to read all these files in a common Python framework. 

There is also a lack of a common analysis tool fostering reproducible analysis of such track data. 
For this reason, HuracanPy's second purpose is to offer a suite of functions to build analysis workflows. 

As such, HuracanPy is mainly an analysis tool that comes after the tracking and up to preliminary plots in a research project. 
Our aim is to make track data analysis more accessible, and to promote good reproducibility practices. 

## Comparison with other packages
HuracanPy aims at becoming a standard community tool covering track data reading and analysis, which did not exist before. 
A few packages covered part of HuracanPy's functionality, with less flexibility in terms of supported track formats and analysis functions:

* Tropycal [@tropycal] is a Python package designed for analysing and visualising tropical cyclone tracks from observations and operational forecasts. However, it does not have the flexibility to read data from other sources. For example, the loading of track data is part of the initialisation of the custom class used by Tropycal. We see HuracanPy as complementary to Tropycal and a future plan is to support conversion of track data to Tropycal objects with HuracanPy.
* The Cyclone Metrics Package (CyMeP) [@zarzycki2021metrics] is a software suite providing a command-line function to provide standard assessment graphs. It can only run with CSV track data, and while it provides a good tool for rapid comparison of datasets, it is not flexible enough for exploratory and in-depth scientific analysis that HuracanPy wants to support. An assessment package based on merging functionalities from CyMeP and HuracanPy is in development. 

# Description

HuracanPy is built on Xarray. It loads track data as an Xarray dataset, and provides an Xarray accessor to provide cyclone-specific analysis methods. 
This allows us to draw from Xarray's performance and existing methods, and also means that we can handle track attributes which are not just scalar.

## Loading data

HuracanPy currently supports loading tracks from:

* CSV (comma-separated values) files where each row corresponds to a cyclone point and each column to a feature.  
* NetCDF (Network Common Data Form) files following CF (Climate and Forecast) conventions [@cfconventions], specifically the formats for trajectory data described in H.4 [@cfconventionswebpage].
* NetCDF files similar to the CF conventions can also be loaded, provided they use specific naming for the track ID.
* Text files from TempestExtreme's StitchNodes (GFDL format, see [@ullrich2021tempestextremes])
* Text files from TRACK [@TRACKa; @TRACKb]
* Text files from IRIS [@sparks2024imperial]
* Text files using the "original HURDAT" format [here](https://www.aoml.noaa.gov/hrd/data_sub/hurdat.html) and used in particular within the European Centre for Medium-range Weather Forecast (ECMWF).

HuracanPy can also load IBTrACS [@gahtan2024international] from an online source, or from an embedded file. 

Tracks are loaded as an Xarray dataset with one main dimension called `record`. 

HuracanPy can also save the track data as CSV and NetCDF files. 

## Analysis

Most of HuracanPy's analysis functions can be called through an Xarray accessor named `hrcn`.

* Manipulating data:
    Loading the track data as an Xarray object means the user can easily use Xarray's methods such as `.where` to subset. 
However, we found it useful to add a specific function that allows for subsetting based on a specific track ID, and on some track properties. 
We also added a specific function to interpolate the track to a different time resolution. 
* Adding track info:
    HuracanPy includes functions to add common geographical and categorical information to the tracks, such as whether a point is over land or the ocean, or which category it belongs to in a defined convention. 
* Computing common diagnoses:
    HuracanPy includes functions to compute track duration, genesis or lifetime maximum intensity attributes, translation speed, lifecycles and rates.
* Tropical Cyclones-specific diagnoses:
    These include accumulated cyclone energy ([ACE](https://en.wikipedia.org/wiki/Accumulated_cyclone_energy)), pressure-derived ACE (PACE) [@zarzycki2021metrics], [Saffir-Simpson categories](https://www.nhc.noaa.gov/aboutsshws.php), Klotzbach pressure categories [@klotzbach2020surface].
* Comparing two (or more) sets of tracks:
    HuracanPy provides a one-on-one track matching function.
    From such a matching table, scores of Probability of Detection (also called Hit Rate) and False Alarm Rate can be computed.
    We also provide a function for computing the delay in onset and offset between the matched tracks.
* Plot:
    HuracanPy currently embeds a small number of functions, which are meant mostly for preliminary visualisation.
    These include plotting the track points and the track density.
    Due to the package being built on Xarray, Xarray's plot function can be used.
    We leave it to the user to build more elaborate plots using their preferred libraries, noting that Xarray in particular works well with seaborn. 

Where possible, functions have been made unit aware by using the accessor and wrapper from MetPy [@metpy], which converts the inputs into a Pint quantity [@pint] internally using the units from the Xarray variable's attributes.

# Perspectives

We welcome suggestions of other track data formats to support, and are happy to receive contributions of code to do so. 

At the moment, HuracanPy only include tropical cyclones--specific diagnoses, but we are more than happy to receive contributions and feedback from other communities, including but not restricted to extra-tropical cyclones, polar lows, medicanes, subtropical cyclones, etc.
While the motivation for HuracanPy is analysis of cyclone tracks, the data format and workflow should also apply to other types of trajectories, such as Lagrangian air-mass trajectories, and we would be interested in supporting common data formats for these.
Main avenues for a potential v2 are supporting multi-dimensional data (e.g. snapshots from TempestExtremes' NodeFileCompose [@ullrich2021tempestextremes], WiTRACK footprints [@befort2020objective]).


# Acknowledgements

Both authors acknowledge financial support from the HUrricane Risk Amplification and Changing North Atlantic Natural disasters (Huracán) NERC-NSF large grant n°NE/W009587/1 (NERC) & AGS-2244917 (NSF). The package is named after this project.
This work also benefited from the TROPICANA program, supported by the Institut Pascal at Université Paris-Saclay, under “Investissements d’avenir” ANR-11-IDEX-0003- 01.

The package includes code that was originally developed in the scope of SB's thesis, which was funded by the Commissariat à l'Energie Atomique et aux Energies Alternatives (CEA), and the EUR IPSL-Climate Graduate School through the ICOCYCLONES2 project, managed by the ANR under the “Investissements d’avenir” programme with the reference ANR-11-IDEX-0004 - 17-EURE-0006.

# References
