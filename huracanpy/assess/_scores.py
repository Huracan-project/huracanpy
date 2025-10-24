"""Functions for score computation"""

import numpy as np


def pod(matches, ref, ref_name):
    """Probability of Detection

    Parameters
    ----------
    matches : pandas.DataFrame
        The result from matching tracks to a reference dataset output from
        :func:`huracanpy.assess.match`
    ref : xarray.Dataset
        The original reference dataset before matching
    ref_name : str
        The name of the reference dataset in `matches`

    Returns
    -------
    float

    """
    n_detected = matches["id_" + ref_name].nunique()
    n_total = len(np.unique(ref.track_id.values))
    return n_detected / n_total


def far(matches, detected, detected_name):
    """False Attribution Rate

    Parameters
    ----------
    matches : pandas.DataFrame
        The result from matching tracks to a reference dataset output from
        :func:`huracanpy.assess.match`
    detected : xarray.Dataset
        The original dataset that was being matched to the reference
    detected_name : str
        The name of the original dataset in `matches`

    Returns
    -------
    float

    """
    n_matched = matches["id_" + detected_name].nunique()
    n_total = len(np.unique(detected.track_id.values))
    return 1 - (n_matched / n_total)
