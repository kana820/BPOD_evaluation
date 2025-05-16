from typing import NamedTuple


class Calibration(NamedTuple):
    # Focal length
    fx: float
    fy: float

    # Principal point
    cx: float
    cy: float

    # Radial distortion
    k1: float
    k2: float

    # Tangential distortion
    p1: float
    p2: float

    # Resolution
    width: int
    height: int