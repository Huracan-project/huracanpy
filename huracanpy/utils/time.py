"""
Utils related to time
"""

def get_time(year, month, day, hour):
    """
    Get np.datetime64 array corresponding to year, month, day and hour arrays

    Parameters
    ----------
    year (np.array or pd.Series)
    month (np.array or pd.Series)
    day (np.array or pd.Series)
    hour (np.array or pd.Series)

    Returns
    -------
    np.array or pd.Series
        The corresponding np.datetime64
    """
    time = pd.to_datetime(
        year.astype(str)
        + "-"
        + month.astype(str)
        + "-"
        + day.astype(str)
        + " "
        + hour.astype(str)
        + ":00"
    )
    return time