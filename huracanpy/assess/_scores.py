"""Functions for score computation"""

import numpy as np


def pod(matches, ref, ref_name):
    """Probability of Detection

    Parameters
    ----------
    matches : pandas.DataFrame
        The result from matching tracks to a reference dataset output from
        `huracanpy.assess.match`
    ref : xarray.Dataset
        The original reference dataset before matching
    ref_name : str
        The name of the reference dataset in `matches`

    Returns
    -------
    float

    """
    N_detected = matches["id_" + ref_name].nunique()
    N_total = len(np.unique(ref.track_id.values))
    return N_detected / N_total


def far(matches, detected, detected_name):
    """False Attribution Rate

    Parameters
    ----------
    matches : pandas.Dataframe
        The result from matching tracks to a reference dataset output from
        `huracanpy.assess.match`
    detected : xarray.Dataset
        The original dataset that was being matched to the reference
    detected_name : str
        The name of the original dataset in `matches`

    Returns
    -------
    float

    """
    N_matched = matches["id_" + detected_name].nunique()
    N_total = len(np.unique(detected.track_id.values))
    return 1 - (N_matched / N_total)
