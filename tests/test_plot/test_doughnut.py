from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import huracanpy


def test_doughnut_basic():
    huracanpy.plot.doughnut(range(20), 10)


def test_doughnut_roberts2020():
    # From the examples notebook
    # Create the input data as you might expect to produce it with huracanpy/xarray
    # A 2d array of average number of storms per year as a function of model and basin
    da = xr.DataArray(
        data=[
            np.asarray([16, 53, 19, 5, 7]) * 0.521,
            np.asarray([17, 46, 23, 7, 7]) * 0.873,
            np.asarray([16, 46, 16, 17, 5]) * 0.509,
            np.asarray([17, 46, 18, 15, 5]) * 0.581,
            np.asarray([13, 45, 20, 18, 4]) * 0.249,
            np.asarray([14, 44, 20, 18, 4]) * 0.42,
            np.asarray([22, 42, 12, 18, 5]) * 0.662,
            np.asarray([23, 44, 9, 20, 4]) * 0.654,
            np.asarray([21, 33, 19, 22, 5]) * 0.14,
            np.asarray([19, 35, 19, 22, 4]) * 0.133,
            np.asarray([11, 32, 21, 28, 8]) * 0.296,
            np.asarray([14, 39, 21, 18, 8]) * 0.649,
            np.asarray([17, 40, 28, 11, 4]) * 0.648,
            np.asarray([19, 39, 26, 10, 6]) * 0.627,
            np.asarray([18, 45, 20, 12, 5]) * 0.495,
            np.asarray([18, 42, 23, 13, 4]) * 0.617,
            np.asarray([21, 38, 24, 11, 6]) * 0.729,
            np.asarray([21, 42, 28, 8, 1]) * 0.536,
        ],
        coords=dict(
            model=[
                "HadGEM3-GC31-LM",
                "HadGEM3-GC31-HM",
                "ECMWF-IFS-LR",
                "ECMWF-IFS-HR",
                "EC-Earth3P-LR",
                "EC-Earth3P-HR",
                "CNRM-CM6-1",
                "CNRM-CM6-1-HR",
                "MPI-ESM1-2-HR",
                "MPI-ESM1-2-XR",
                "CMCC-CM2-HR4",
                "CMCC-CM2-VHR4",
                "MERRA2",
                "JRA55",
                "ERAI",
                "ERA5",
                "CFSR2",
                "Obs",
            ],
            basin=["na", "wp", "ep", "ni", "other"],
        ),
    )

    # Specific parameters to plt.pie used by Roberts et al. (2020). See
    # https://github.com/eerie-project/storm_track_analysis/blob/main/assess/tc_assessment.py#L384
    pie_kwargs = dict(
        startangle=90,
        pctdistance=0.7,
        autopct="%1.0f%%",
        labels=da.basin.values,
        labeldistance=1.0,
        colors=["#ff6666", "#ffcc99", "#cc9966", "#cc6666", "#66b3ff"],
    )

    # The second value in the centre is the number of southern hemisphere storms
    # This isn't related to the data in the doughnut so I've just put a list of values
    # here
    sh_values = [
        68.3,
        95.0,
        48.4,
        53.9,
        33.4,
        39.3,
        64.3,
        60.1,
        20.9,
        20.3,
        24.2,
        48.5,
        53.9,
        48.3,
        46.0,
        51.5,
        53.3,
        20.8,
    ]

    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    axes = axes.flatten()

    # Thickness of doughnuts relative to the "Obs" doughnut
    reference_total = da.sel(model="Obs").values.sum()

    # One plot for each model. Loop over array per model
    for n, model in enumerate(da.model.values):
        da_ = da.sel(model=model)

        huracanpy.plot.doughnut(da_.values, reference_total, ax=axes[n], **pie_kwargs)
        axes[n].text(
            0, 0, f"{da_.values.sum():.1f}\n{sh_values[n]}", ha="center", va="center"
        )
        axes[n].set_title(f"({ascii_lowercase[n]}) {model}")
