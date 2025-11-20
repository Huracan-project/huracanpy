"""
Module defining objects for TC-related conventions
"""

import numpy as np
from metpy.units import units

# Categories
_thresholds = {
    "Beaufort": dict(
        bins=np.asarray(
            [
                -np.inf,
                1,
                3.5,
                6.5,
                10.5,
                16.5,
                21.5,
                27.5,
                33.5,
                40.5,
                47.5,
                55.5,
                63.5,
                np.inf,
            ]
        )
        * units("knots"),
        labels=list(range(12 + 1)),
    ),
}
