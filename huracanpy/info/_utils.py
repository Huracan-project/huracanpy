import numpy as np


def inferred_track_id(*variables):
    """Create a track_id variable by combining multiple identifying variables

    If, for example, we have a set of tracks with variables `year` and `storm_number`,
    but `storm_number` is reset to zero for each year, we can still uniquely identify
    each track using the combination of `year` and `storm_number`. This function
    does that identification and creates a track_id array, e.g.

    >>> track_id = huracanpy.inferred_track_id(tracks.year,tracks.storm_number)

    Parameters
    ----------
    *variables
        Pass a number of variables required to determine the unique track_id

    Returns
    -------
    array_like
        An array of integers ranging from 0 to the total number of tracks. Identifying
        the track of each record. The return type is the same type as the input or a
        numpy.ndarray if the input does not have a `copy` method

    """
    long_id = ""
    for variable in variables:
        _, partial_id = np.unique(variable, return_inverse=True)

        # Split up the individual IDs to avoid accidental matches
        # e.g. (11, 1) and (1, 11) are 11-1 and 1-11 rather than both 111
        long_id = np.char.add(long_id, "-")
        long_id = np.char.add(long_id, partial_id.astype(str))

    _, track_id = np.unique(long_id, return_inverse=True)

    # Try to return the same type as the input
    try:
        track_id_return = variables[0].copy()
        track_id_return[:] = track_id

        return track_id_return

    except AttributeError:
        return track_id
