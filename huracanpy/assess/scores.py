"""Functions for score computation"""

import numpy as np


def POD(matches, ref, ref_name="ib"):
    N_detected = matches["id_" + ref_name].nunique()
    N_total = len(np.unique(ref.track_id.values))
    return N_detected / N_total


def FAR(matches, detected, detected_name="UZ"):
    N_matched = matches["id_" + detected_name].nunique()
    N_total = len(np.unique(detected.track_id.values))
    return 1 - (N_matched / N_total)
