# Non-huracanpy functions shared by various functions
def combine_kws(kws, kws_default):
    # Generic function to check on an optional dictionary argument and merge it with
    # default arguments
    if kws is None:
        return kws_default.copy()
    # Overwrite default arguments with explicit arguments
    return {**kws_default, **kws}
