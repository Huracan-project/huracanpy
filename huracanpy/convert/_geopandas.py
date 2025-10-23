from cartopy.crs import Geodetic
from geopandas import GeoDataFrame
import numpy as np
from shapely.geometry import Point, LineString


def to_geodataframe(lon, lat, track_id=None, *, crs=None):
    if crs is None:
        crs = Geodetic()

    xyz = Geodetic().transform_points(crs, lon, lat)

    # Geodetic transform treats -180 as different to 180 which can lead to a wrap around
    # the globe when it should be not moving. Use -180 as an arbitrary convention
    xyz[:, 0][xyz[:, 0] == 180] = -180

    # Convert tracks to a dictionary with lon, lat points as a geometry
    # Create the Shapely geometry from lon, lat points
    if track_id is None:
        tracks_dict = dict(geometry=[Point(xy) for xy in xyz[:, :2]])

    else:
        track_id = np.asarray(track_id)

        # Split dateline crossings into two lines
        dateline_crossing = np.where(
            (np.abs(np.diff(xyz[:, 0])) > 180) & (track_id[1:] == track_id[:-1])
        )[0]

        # Track offset as elements get input into the array
        for offset, idx in enumerate(dateline_crossing):
            # np.insert puts the new value in front of the index so add 1
            loc = idx + 2 * offset + 1
            track_id = np.insert(track_id, loc, track_id[loc])
            track_id = np.insert(track_id, loc, track_id[loc + 1])

            y_mid = 0.5 * (xyz[loc - 1, 1] + xyz[loc, 1])

            # Order of insertion depends on direction of dateline crossing
            if xyz[loc - 1, 0] > 0:
                xyz = np.insert(xyz, loc, np.array([180 - 1e-7, y_mid, 0]), axis=0)
                xyz = np.insert(xyz, loc + 1, np.array([-180, y_mid, 0]), axis=0)
            else:
                xyz = np.insert(xyz, loc, np.array([-180, y_mid, 0]), axis=0)
                xyz = np.insert(xyz, loc + 1, np.array([180 - 1e-7, y_mid, 0]), axis=0)

            # Update the location of the dateline crossing to account for the inserted
            # points
            dateline_crossing[offset] = loc

        # All the indices for the start of distinct linestrings
        # Include the start and end indices
        indices = np.sort(
            np.concatenate(
                [
                    dateline_crossing + 1,
                    np.where(track_id[1:] != track_id[:-1])[0] + 1,
                    [0, len(track_id)],
                ]
            )
        )

        tracks_dict = dict(
            track_id=track_id[indices[:-1]],
            geometry=[
                LineString(xyz[indices[n] : indices[n + 1], :2])
                if (indices[n + 1] - indices[n]) > 1
                else Point(xyz[indices[n], :2])
                for n in range(len(indices) - 1)
            ],
        )

    # Return as a GeoDataFrame
    return GeoDataFrame(tracks_dict, crs=Geodetic())
