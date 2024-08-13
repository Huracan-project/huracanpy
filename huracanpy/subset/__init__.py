__all__ = ["trackswhere", "sel_id"]

from ._where import trackswhere


def sel_id(data, tid, track_id_name="track_id"):
    df = data.to_dataframe()
    track = df[df[track_id_name] == tid]
    return track.to_xarray()
